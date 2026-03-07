import csv
from PIL import Image

# 1. Update these paths as needed
img_path = "image.png"
output_csv = "pixel_data.csv"

# Open and ensure image is in RGB mode
img = Image.open(img_path).convert("RGB")
width, height = img.size
pixels = img.load()

# Define the header for the CSV
header = ["no", "x", "y", "r", "g", "b"]

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    pixel_count = 0

    # Iterate through every pixel
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            pixel_count += 1

            # Create a flat row: [no, x, y, r, g, b]
            writer.writerow([pixel_count, x, y, r, g, b])

print(f"Success! {pixel_count} pixels exported to '{output_csv}'.")
