
import torch
from datasets import FER2013
from torch.utils.data import DataLoader
from torchvision import transforms
import config as cnf
from tqdm import tqdm
from model import Net
from tensorboardX import SummaryWriter


def create_dataloaders(batch_size=64):
    # transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cnf.IMAGE_SIZE, cnf.IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    train_dataset = FER2013('fer2013', train=True, transform=transform)
    val_dataset = FER2013('fer2013', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)
    return train_loader, val_loader

def train(model, dataloader, criterion, optim):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def valid(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

if __name__ == '__main__':
    writer = SummaryWriter(log_dir='log/source/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_dataloaders(cnf.SOURCE_BATCH_SIZE)
    
    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0
    step = 0
    
    for epoch in range(1, cnf.SOURCE_EPOCH+1):
        train_loss, train_accuracy = train(model, train_loader, criterion, optim=optim)
        writer.add_scalar('Train Loss', float(train_loss), step)
        writer.add_scalar('Train Accuracy', float(train_accuracy), step)
    
        with torch.no_grad():
            val_loss, val_accuracy = valid(model, val_loader, criterion)
            writer.add_scalar('Validation Loss', float(val_loss), step)
            writer.add_scalar('Validation Accuracy', float(val_accuracy), step)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), cnf.TRAINED_MODEL_PATH)

        lr_schedule.step(val_loss)
        step+=1

