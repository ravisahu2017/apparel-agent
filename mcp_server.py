import os
from mcp.server.fastmcp import FastMCP
from extractor_chain import VisionExtractorChain
from generater_chain import GeneratorChain

# Initialize FastMCP server
mcp = FastMCP("Apparel-Designer")

@mcp.tool()
def extract_apparel_design(image_paths: list[str], product_id: str) -> dict:
    """
    Extracts design DNA from a apparel image (Silhouette, Fabric, Motif).
    Uses Claude-3-Haiku via OpenRouter.
    """

    print(f"Extracting design DNA for product {product_id} with {len(image_paths)} images")

    vision_chain = VisionExtractorChain(tinydb_path=f"db/products.nogit.json")
    design_dna = vision_chain.invoke({"image_paths": image_paths, "product_id": product_id})
    
    return design_dna

@mcp.tool()
def generate_prompt_for_view(product_id: str, design_json: dict, view: str, market_place: str = "Meesho") -> str:
    """
    Generates a prompt for a specific view of a product.
    Input:
    - product_details: dict containing product information
    - view: str, the view of the product (e.g., "front", "back", "detail")
    - market_place: str, the market place (e.g., "Meesho", "Amazon", "Flipkart")
    """
    # Logic to call ModelFactory.call_model("generation", ...)

    generator_chain = GeneratorChain(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
    )

    gen_result = generator_chain.generate_prompt(
        inputs={
            "product_id": product_id,
            "design_json": design_json,
            "description": f"""
            Generate a {view} of the modal in the mentioned kurti
            [Input image details]: 
            - First image is the front view of the garment
            - Second image is the back view of the garment
            - Third image is the neckline detail view of the garment
            - Fourth image is showing the closeup of the repeat Pattern
            """,
            "view": view,
            "market_place": market_place,
        }
    )

    return gen_result


@mcp.tool()
def generate_image_from_prompt(prompt: str, input_images: list[str]) -> str:
    """
    Generates an image from a prompt.
    Input:
    - prompt: str, the prompt to generate the image from
    - input_images: list[str], the input images to use for the generation
    """
    
    generator_chain = GeneratorChain(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENROUTER_API_KEY")
    )
    os.makedirs("output", exist_ok=True)

    gen_result = generator_chain.generate_image_v2(prompt, input_images)

    return gen_result


if __name__ == "__main__":
    mcp.run(transport="sse")