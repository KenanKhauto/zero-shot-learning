import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel

class ClipCifar:
    def __init__(self, dataset_root='./datasets', batch_size=32, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.dataset_root = dataset_root
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.text_inputs = [f"A photo of a {c}" for c in self.classes]

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.CIFAR10(root=self.dataset_root, train=False, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def classify_images(self):
        self.model.eval()
        correct_predictions = 0
        total_images = 0

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                inputs = self.processor(text=self.text_inputs, 
                                        images=images, 
                                        return_tensors="pt", 
                                        padding=True, 
                                        truncation=True,
                                        ).to(self.device)
                
                
                
                text_emb = self.model.get_text_features(**{k: inputs[k] for k in inputs.keys() - {"pixel_values"}})
                text_emb = text_emb / torch.norm(text_emb, dim=0)

                img_emb = self.model.get_image_features(**{k: inputs[k] for k in inputs.keys() & {"pixel_values"}}
                                                        )
                
                similarities = img_emb @ text_emb.T
                pred = torch.argmax(similarities, dim=1)
                correct_predictions += (pred == labels.to(self.device)).sum().item()
                total_images += labels.size(0)
        
                # outputs = self.model(**inputs)
                # # Get the logits
                # logits_per_image = outputs.logits_per_image  # Image-to-text similarity
                # predicted_classes = logits_per_image.argmax(dim=-1)

                # correct_predictions += (predicted_classes == labels.to(self.device)).sum().item()
                # total_images += labels.size(0)

        accuracy = correct_predictions / total_images
        return accuracy


if __name__ == "__main__":
    model = ClipCifar()
    model.load_dataset()
    accuracy = model.classify_images()
    print(accuracy)