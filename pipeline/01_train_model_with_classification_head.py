# General imports
import os
import sys
from pathlib import Path
import ast
import pandas as pd

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter

# Internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../pdl1_project/prov-gigapath'))

from datasets.HEIHCDataset import HEIHCDataset
from gigapath.pipeline import load_tile_slide_encoder
from losses.tangle_loss import InfoNCE, apply_random_mask
from utils.process_args import process_args
from utils.learning import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSL(nn.Module):
    def __init__(self, num_classes=2):
        super(SSL, self).__init__()
        _, slide_encoder = load_tile_slide_encoder(global_pool=True)
        self.slide_encoder = slide_encoder

        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),  # Reduce dimension
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)  # Map to number of classes
        )
    
    def forward(self, tile_embeds, coords):
        if len(tile_embeds.shape) == 2:
            tile_embeds = tile_embeds.unsqueeze(0)
            coords = coords.unsqueeze(0)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            slide_embeds = self.slide_encoder(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)    

        outputs = {"layer_{}_embed".format(i): slide_embeds[i] for i in range(len(slide_embeds))}
        outputs["last_layer_embed"] = slide_embeds[-1]

        logits = self.classification_head(slide_embeds[-1])
        outputs["logits"] = logits

        return outputs
    
def train_loop(args, loss_fn_interMod, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, loss_fn_classification):

    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss, ep_inter_loss, ep_intra_loss, ep_class_loss = 0., 0., 0., 0.

    for b_idx, (he_patch_emb, ihc_patch_emb, he_patch_coords, ihc_patch_coords, labels) in enumerate(dataloader):
        losses = []

        he_patch_emb = he_patch_emb.to(DEVICE)
        ihc_patch_emb = ihc_patch_emb.to(DEVICE)
        he_patch_coords = he_patch_coords.to(DEVICE)
        ihc_patch_coords = ihc_patch_coords.to(DEVICE)
        labels = labels.to(DEVICE)
        
        ## intra modality loss H&E <-> H&E
        if not args['only_class_loss']:
            he_patch_emb_mask = apply_random_mask(he_patch_emb, 0.5)

            # # slide embeddings with and without mask
            HandE_wsi_emb = ssl_model(he_patch_emb, he_patch_coords)
            HandE_wsi_emb_mask = ssl_model(he_patch_emb_mask, he_patch_coords)

            # adding intraMod loss
            losses.append(loss_fn_intraMod(HandE_wsi_emb['last_layer_embed'], HandE_wsi_emb_mask['last_layer_embed']))
            ep_intra_loss += losses[-1].item()

            ## inter modality loss H&E <-> IHC
            IHC_wsi_emb = ssl_model(ihc_patch_emb, ihc_patch_coords)
            losses.append(loss_fn_interMod(HandE_wsi_emb['last_layer_embed'], IHC_wsi_emb['last_layer_embed'], symmetric=True))
            ep_inter_loss += losses[-1].item()

        ## Classification loss
        logits = ssl_model(he_patch_emb, he_patch_coords)["logits"]
        class_loss = loss_fn_classification(logits, labels)
        losses.append(class_loss)
        ep_class_loss += class_loss.item()
        
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch <= args["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")
            
        ep_loss += loss.item()

    return ep_loss / len(dataloader), ep_intra_loss / len(dataloader), ep_inter_loss / len(dataloader), ep_class_loss / len(dataloader)

def val_loop(args, loss_fn_interMod, loss_fn_intraMod, ssl_model, dataloader, loss_fn_classification):
    ssl_model.eval()  # Set model to evaluation mode
    ssl_model.to(DEVICE)

    ep_loss, ep_inter_loss, ep_intra_loss, ep_class_loss = 0., 0., 0., 0.
    with torch.no_grad():  # Disable gradient computation
        for b_idx, (he_patch_emb, ihc_patch_emb, he_patch_coords, ihc_patch_coords, labels) in enumerate(dataloader):
            losses = []

            # Move data to device
            he_patch_emb = he_patch_emb.to(DEVICE)
            ihc_patch_emb = ihc_patch_emb.to(DEVICE)
            he_patch_coords = he_patch_coords.to(DEVICE)
            ihc_patch_coords = ihc_patch_coords.to(DEVICE)
            labels = labels.to(DEVICE)

            if not args['only_class_loss']:
                ## Intra-modality loss H&E <-> H&E
                he_patch_emb_mask = apply_random_mask(he_patch_emb, 0.5)

                # Slide embeddings with and without mask
                HandE_wsi_emb = ssl_model(he_patch_emb, he_patch_coords)
                HandE_wsi_emb_mask = ssl_model(he_patch_emb_mask, he_patch_coords)

                # Add intraMod loss
                intra_loss = loss_fn_intraMod(HandE_wsi_emb['last_layer_embed'], HandE_wsi_emb_mask['last_layer_embed'])
                losses.append(intra_loss)
                ep_intra_loss += intra_loss.item()

                ## Inter-modality loss H&E <-> IHC
                IHC_wsi_emb = ssl_model(ihc_patch_emb, ihc_patch_coords)
                inter_loss = loss_fn_interMod(HandE_wsi_emb['last_layer_embed'], IHC_wsi_emb['last_layer_embed'], symmetric=True)
                losses.append(inter_loss)
                ep_inter_loss += inter_loss.item()

            # Classification loss
            logits = ssl_model(he_patch_emb, he_patch_coords)["logits"]
            class_loss = loss_fn_classification(logits, labels)
            losses.append(class_loss)
            ep_class_loss += class_loss.item()

            # Accumulate total loss
            loss = sum(losses)
            ep_loss += loss.item()

    return ep_loss / len(dataloader), ep_intra_loss / len(dataloader), ep_inter_loss / len(dataloader), ep_class_loss / len(dataloader)

def get_matching_pairs_p53(df_info):
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        matching_p53_list = ast.literal_eval(row['Matching p53'])
        if matching_p53_list:
            IHC_slides.append(matching_p53_list[0].split(".")[0])
            HE_slides.append(row['svs_file'].split(".")[0])
    return list(zip(HE_slides, IHC_slides))

def get_matching_pairs_pdl1(df_info):
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        he_slide, ihc_slide = row['he_slide_name'].split(".")[0], row['ihc_slide_name'].split(".")[0]

        # check if tile embeddings are present
        he_path = DATA_DIR / f"{he_slide}.h5"
        ihc_path = DATA_DIR / f"{ihc_slide}.h5"

        if not he_path.exists() or not ihc_path.exists():
            print(f"[INFO] Missing tile embeddings for slide {he_slide} or {ihc_slide}. Skipping...")
            continue

        HE_slides.append(he_slide)
        IHC_slides.append(ihc_slide)
    return list(zip(HE_slides, IHC_slides))

def get_matching_pairs_ki67(df_info):
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        he_slide, ihc_slide = row['HE_Slide'].split(".")[0], row['IHC_Slide'].split(".")[0]

        # check if tile embeddings are present
        he_path = DATA_DIR / f"{he_slide}.h5"
        ihc_path = DATA_DIR / f"{ihc_slide}.h5"

        if not he_path.exists() or not ihc_path.exists():
            print(f"[INFO] Missing tile embeddings for slide {he_slide} or {ihc_slide}. Skipping...")
            continue

        HE_slides.append(he_slide)
        IHC_slides.append(ihc_slide)
    return list(zip(HE_slides, IHC_slides))

if __name__ == "__main__":
    args = process_args()
    args = vars(args)
    set_seed(args["seed"])

    if args['only_class_loss']:
        EXP_ID = (
            f"tangle_{args['study']}_binary_"
            f"{args['n_tokens']}_{args['temperature']}_"
            f"{args['learning_rate']}_{args['end_learning_rate']}_"
            f"{args['batch_size']}_{args['epochs']}_class_head_only_class_new"
        )
    else:
        EXP_ID = (
            f"tangle_{args['study']}_binary_"
            f"{args['n_tokens']}_{args['temperature']}_"
            f"{args['learning_rate']}_{args['end_learning_rate']}_"
            f"{args['batch_size']}_{args['epochs']}_class_head_new"
        )

    folds = [i for i in range(args["folds"])]
    for fold in folds:
        RESULTS_SAVE_PATH = Path(f"{EXP_ID}/fold_{fold}")
        DATA_DIR = Path(args['data_dir'])
        DF_INFO = pd.read_csv(args['info_csv']) 

        writer = SummaryWriter(log_dir=os.path.join(RESULTS_SAVE_PATH, "tensorboard_logs"))

        ssl_model = SSL().to(DEVICE)

        gpu_devices = [int(gpu) for gpu in args["gpu_devices"].split(",")]
        if len(gpu_devices) > 1:
            print(f"* Using {torch.cuda.device_count()} GPUs.")
            ssl_model = nn.DataParallel(ssl_model, device_ids=gpu_devices)
        ssl_model.to("cuda:0")

        print("* Setup optimizer...")
        optimizer = optim.AdamW(ssl_model.parameters(), lr=args["learning_rate"])

        # handling specific to each stain
        print("* Setup dataloaders...")
        if args['study'] == 'pdl1':
            # FOR TRAINING
            print(f"[INFO] Reading from {args['dataset_csv']}/train_{fold}.csv")
            df_info_train = pd.read_csv(f"{args['dataset_csv']}/train_{fold}.csv")
            df_info_train = df_info_train.rename(columns={'slide_id': 'he_slide_name'})
            df_info_train = pd.merge(df_info_train, DF_INFO[['he_slide_name', 'ihc_slide_name']], on='he_slide_name', how='left')
            HE_IHC_pairs_train = get_matching_pairs_pdl1(df_info_train)
            dataset_train = HEIHCDataset(DATA_DIR, HE_IHC_pairs_train, args["n_tokens"], df_info_train, args['study'])
            # FOR VALIDATION
            print(f"[INFO] Reading from {args['dataset_csv']}/val_{fold}.csv")
            df_info_val = pd.read_csv(f"{args['dataset_csv']}/val_{fold}.csv")
            df_info_val = df_info_val.rename(columns={'slide_id': 'he_slide_name'})
            df_info_val = pd.merge(df_info_val, DF_INFO[['he_slide_name', 'ihc_slide_name']], on='he_slide_name', how='left')
            HE_IHC_pairs_val = get_matching_pairs_pdl1(df_info_val)
            dataset_val = HEIHCDataset(DATA_DIR, HE_IHC_pairs_val, args['n_tokens'], df_info_val, args['study'])
        elif args['study'] == 'p53':
            # FOR TRAINING
            df_info_train = pd.read_csv(f"{args['dataset_csv']}/train_{fold}.csv")
            df_info_train = df_info_train.rename(columns={'slide_id': 'svs_file'})
            df_info_train = pd.merge(df_info_train, DF_INFO[['svs_file', 'Matching p53']], on='svs_file', how='left')
            HE_IHC_pairs_train = get_matching_pairs_p53(df_info_train)
            dataset_train = HEIHCDataset(DATA_DIR, HE_IHC_pairs_train, args["n_tokens"], df_info_train, args['study'])
            # FOR VALIDATION
            df_info_val = pd.read_csv(f"{args['dataset_csv']}/val_{fold}.csv")
            df_info_val = df_info_val.rename(columns={'slide_id': 'svs_file'})
            df_info_val = pd.merge(df_info_val, DF_INFO[['svs_file', 'Matching p53']], on='svs_file', how='left')
            HE_IHC_pairs_val = get_matching_pairs_p53(df_info_val)
            dataset_val = HEIHCDataset(DATA_DIR, HE_IHC_pairs_val, args['n_tokens'], df_info_val, args['study'])
        elif args['study'] == 'ki67':
            # FOR TRAINING
            print(f"[INFO] Reading from {args['dataset_csv']}/train_{fold}.csv")
            df_info_train = pd.read_csv(f"{args['dataset_csv']}/train_{fold}.csv")
            df_info_train = df_info_train.rename(columns={'slide_id': 'HE_Slide'})
            df_info_train = pd.merge(df_info_train, DF_INFO[['HE_Slide', 'IHC_Slide']], on='HE_Slide', how='left')
            HE_IHC_pairs_train = get_matching_pairs_ki67(df_info_train)
            dataset_train = HEIHCDataset(DATA_DIR, HE_IHC_pairs_train, args["n_tokens"], df_info_train, args['study'])
            # FOR VALIDATION
            print(f"[INFO] Reading from {args['dataset_csv']}/val_{fold}.csv")
            df_info_val = pd.read_csv(f"{args['dataset_csv']}/val_{fold}.csv")
            df_info_val = df_info_val.rename(columns={'slide_id': 'HE_Slide'})
            df_info_val = pd.merge(df_info_val, DF_INFO[['HE_Slide', 'IHC_Slide']], on='HE_Slide', how='left')
            HE_IHC_pairs_val = get_matching_pairs_ki67(df_info_val)
            dataset_val = HEIHCDataset(DATA_DIR, HE_IHC_pairs_val, args['n_tokens'], df_info_val, args['study'])

        train_dataloader = DataLoader(dataset_train, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
        val_dataloader = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=False)
        print(f"Number of training samples: {len(dataset_train)}")

        # set up schedulers
        print("* Setup schedulers...")
        T_max = (args["epochs"] - args["warmup_epochs"]) * len(train_dataloader) if args["warmup"] else args["epochs"] * len(train_dataloader)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=T_max,
            eta_min=args["end_learning_rate"]
        )
        
        if args["warmup"]:
            scheduler_warmup = LinearLR(
                optimizer, 
                start_factor=0.00001,
                total_iters=args["warmup_epochs"] * len(train_dataloader)
            )
        else:
            scheduler_warmup = None

        # set up losses
        print("* Setup losses...")
        loss_fn_classification = nn.CrossEntropyLoss()

        if not args['only_class_loss']:
            loss_fn_intraMod = nn.MSELoss()
            loss_fn_interMod = InfoNCE(temperature=args["temperature"])
        else:
            loss_fn_intraMod = None
            loss_fn_interMod = None
    
        # main training loop
        min_val_loss = float('inf')
        for epoch in range(args["epochs"]):
            
            print()
            print(f"Training for epoch {epoch}...")
            print()
            
            ep_train_loss, ep_train_intra_loss, ep_train_inter_loss, ep_train_class_loss = train_loop(args, loss_fn_interMod, loss_fn_intraMod, 
                                                                                 ssl_model, epoch, train_dataloader, optimizer, scheduler_warmup, scheduler, loss_fn_classification)
            
            if args['only_class_loss']:
                assert ep_train_intra_loss == 0.0 and ep_train_inter_loss == 0.0, "Intra and Inter losses should be zero for only class loss"
            
            print()
            print(f"Validation for epoch {epoch}...")
            print()

            ep_val_loss, ep_val_intra_loss, ep_val_inter_loss, ep_val_class_loss = val_loop(args, loss_fn_interMod, loss_fn_intraMod, ssl_model, val_dataloader, loss_fn_classification)

            if args['only_class_loss']:
                assert ep_val_intra_loss == 0.0 and ep_val_inter_loss == 0.0, "Intra and Inter losses should be zero for only class loss"

            print()
            print(f"Done with epoch {epoch}")
            print(f"Training = {ep_train_loss}")
            print(f"Validation loss = {ep_val_loss}")

            # Log epoch-wise metrics
            writer.add_scalar("Loss/Epoch_Train", ep_train_loss, epoch)
            writer.add_scalar("Loss/Epoch_Train_Intra", ep_train_intra_loss, epoch)
            writer.add_scalar("Loss/Epoch_Train_Inter", ep_train_inter_loss, epoch)
            writer.add_scalar("Loss/Epoch_Train_Class", ep_train_class_loss, epoch)
            writer.add_scalar("Loss/Epoch_Val", ep_val_loss, epoch)
            writer.add_scalar("Loss/Epoch_Val_Intra", ep_val_intra_loss, epoch)
            writer.add_scalar("Loss/Epoch_Val_Inter", ep_val_inter_loss, epoch)
            writer.add_scalar("Loss/Epoch_Val_Class", ep_val_class_loss, epoch)

            # Log learning rate
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Learning_Rate/Group_{i}", param_group['lr'], epoch)

            # Stop training based on loss of the training samples. Ok for TANGLE and Intra. 
            if ep_val_loss < min_val_loss:
                print('Better loss found. Saving model...')
                min_val_loss = ep_val_loss
                os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
                torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model.pt"))
            print()

            # Save model every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Saving model at epoch {epoch}")
                os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
                torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, f"model_epoch_{epoch}.pt"))
            print()
        
        print()
        print("Done")
        print()

    writer.close()
    