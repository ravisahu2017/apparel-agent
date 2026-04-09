from mcp_server import extract_apparel_design, generate_prompt_for_view, generate_image_from_prompt
from tools.image_util import read_image_files

image_files = read_image_files("input_images")

# product_details = extract_apparel_design(image_files)
# print(product_details)

product_details =  {'design_attributes': {'repeating_pattern.jpg': {'Type of garment': 'fabric sample', 'Silhouette': 'rectangular', 'Patterns': ['floral motifs', 'leaf motifs'], 'Colors': ['red', 'white'], 'Sleeves': 'none', 'Top length': 'N/A', 'Neck design': 'N/A', 'Border hem': 'simple hem', 'Notable visual details': 'intricate embroidery', 'Style category': 'decorative'}, 'fabric.jpg': {'Type of garment': 'kurti', 'Silhouette': 'rectangular (flat fabric)', 'Patterns': ['floral motifs', 'leaf motifs'], 'Colors': ['red', 'cream'], 'Sleeves': 'none', 'Top length': 'N/A', 'Neck design': 'N/A', 'Border hem': 'simple hem', 'Notable visual details': 'detailed embroidery', 'Style category': 'casual'}, 'neck.png': {'Type of garment': 'kurti', 'Silhouette': 'rounded neckline', 'Patterns': ['lace trim', 'button detailing'], 'Colors': ['white'], 'Sleeves': 'N/A', 'Top length': 'Midi length', 'Neck design': 'rounded with lace trim and buttons', 'Border hem': 'N/A', 'Notable visual details': 'lace detailing', 'Style category': 'ethnic'}, 'back.png': {'Type of garment': 'kurti', 'Silhouette': 'rectangular (back view)', 'Patterns': ['floral/leaf motifs (repeating)'], 'Colors': ['red'], 'Sleeves': 'N/A', 'Top length': 'Midi length', 'Neck design': 'simple hem (no lace trim)', 'Border hem': 'N/A', 'Notable visual details': 'same pattern as front, no lace trim', 'Style category': 'ethnic'}, 'front.png': {'Type of garment': 'kurti', 'Silhouette': 'tunic-style (Midi length)', 'Patterns': ['repeating floral/leaf motifs'], 'Colors': ['red', 'white'], 'Sleeves': '3/4th', 'Top length': 'Midi length', 'Neck design': 'rounded with lace trim and button detailing', 'Border hem': 'N/A (Midi hem)', 'Notable visual details': 'intricate embroidery with lace trim', 'Style category': 'casual ethnic'}}, 'product_id': 'b786fcf2-6a52-4dbb-b971-9fd088635785'}
# generated_prompt = generate_prompt_for_view(product_details, "front", "Amazon")

generated_prompt = """
[Primary Reference]: Use the attached source images (front.png, back.png, neck.png, repeating_pattern.jpg, fabric.jpg) as the structural foundation for the kurti.
[Actionable Scene]: A professional female fashion model standing in a front-facing pose, showcasing the kurti in a high‑end Amazon e‑commerce catalog style, studio lighting, clean white background, 8k resolution, photorealistic.
[Garment Integrity Contract]:
    - The model wears the EXACT kurti as depicted in the reference images.
    - Silhouette: tunic‑style midi length with rectangular drape.
    - Sleeves: 3/4 length as shown in front.png.
    - Neck design: rounded neckline with lace trim and button detailing (neck.png).
    - Patterns: repeating floral and leaf motifs in red and white/cream (as per front.png, back.png, fabric.jpg, repeating_pattern.jpg).
    - Colors: primary red base with white/cream floral/leaf embroidery.
    - Border hem: simple midi hem (no additional trim).
    - Notable visual details: intricate embroidery with lace trim on neckline and sleeve cuffs; pineapple‑style motifs within the floral pattern must be sharp and consistent.
    - Style category: casual ethnic.
[Image‑to‑Image Logic]:
Transfer the garment from the reference images onto the model, preserving the exact silhouette, fabric drape, pattern placement, and embroidery detail. Ensure the texture of the fabric, the sharpness of the embroidery edges, and the fidelity of the pineapple/floral motifs match the source images exactly. The final image should reflect a seamless integration of the reference kurti onto the model’s form, with no alterations to design elements."""

result = generate_image_from_prompt(generated_prompt, image_files, product_details)
print(result)


