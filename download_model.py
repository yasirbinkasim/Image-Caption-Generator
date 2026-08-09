from transformers import BlipProcessor, BlipForConditionalGeneration

print("Downloading BLIP Model... Please wait...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ Model successfully downloaded and cached on your PC!")