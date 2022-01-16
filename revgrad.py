import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import FER2013, StudentEmotion
from tqdm import tqdm
import config as cnf
from model import Net
from utils import GradientReversal
from tensorboardX import SummaryWriter

def create_source_dataloader(batch_size, shuffle=True):
    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cnf.IMAGE_SIZE, cnf.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = FER2013('fer2013', train=True, transform=transform)
    val_dataset = FER2013('fer2013', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, val_loader

def create_target_dataloader(batch_size, shuffle=True):
    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cnf.IMAGE_SIZE, cnf.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = StudentEmotion('stu_emotion', transform=transform)
    
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return dataloader

def create_discriminator():
    return nn.Sequential(
            GradientReversal(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        ) 

def main():
    writer = SummaryWriter(log_dir='log/revgrad/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load(cnf.TRAINED_MODEL_PATH))
    feature_extractor = model.feature_extractor
    clf = model.classifier

    discriminator = create_discriminator()
    discriminator.to(device)

    half_batch = cnf.RAVGRAD_BATCH_SIZE // 2
    source_loader, _ = create_source_dataloader(batch_size=half_batch)
    target_loader = create_target_dataloader(half_batch, shuffle=True)
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()), lr=cnf.RAVGRAD_LEARING_RATE)

    step = 0

    for epoch in range(1, cnf.RAVGRAD_EPOCH+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0

        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                features = feature_extractor(x).view(x.shape[0], -1)
                domain_preds = discriminator(features).squeeze()
                label_preds = clf(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')
        
        writer.add_scalar('Domain Loss', float(mean_loss), step)
        writer.add_scalar('Source Accuracy', float(mean_accuracy), step)
        step +=1

        torch.save(model.state_dict(), 'trained_models/revgrad.pt')


if __name__ == '__main__':
    main()