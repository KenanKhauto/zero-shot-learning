import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel

class CLIPcls:
    def __init__(self, dataset_pt, batch_size, use_gpu, classes, prombt_eng):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.dataset = dataset_pt
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.classes = classes
        if prombt_eng:
            self.text_inputs = [f"A photo of a {c}" for c in self.classes]
        else:
            self.text_inputs = classes
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False)

    def classify_images(self):
        self.model.eval()
        correct_predictions = 0
        total_images = 0
        # Precompute text embeddings
        text_inputs = self.processor(text=self.text_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_emb = self.model.get_text_features(**{k: text_inputs[k] for k in text_inputs.keys() if k != "pixel_values"})
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                img_inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                img_emb = self.model.get_image_features(**{k: img_inputs[k] for k in img_inputs.keys() if k == "pixel_values"})
                img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)

                # Calculate similarities
                similarities = img_emb @ text_emb.T
                pred = torch.argmax(similarities, dim=1)

                # Compare with labels and update correct predictions
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
    from utils.utils import read_image
    labels = ["A photo of a piano", 
              "Someone playing the piano", 
              "A photo of a guitar", 
              "A photo of a piano in a white background",
              "A very big dog eating hotdogs", 
              "A fluffy cat", 
              "A photo of the earth from the dark space"]
    
    im1 = read_image(image_name="1.jpg")
    im2 = read_image(image_name="2.jpg")
    im3 = read_image(image_name="3.jpg")
    
    transforms = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    im1 = transforms(im1)
    im2 = transforms(im2)
    im3 = transforms(im3)

    data = torch.stack([im1, im2, im3])

    print(data.shape)
    # model = CLIPcls()
    # accuracy = model.classify_images()
    # print(accuracy)