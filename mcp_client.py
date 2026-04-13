import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def extract_apparel_design(image_paths, product_id):
    # Connect directly to MCP server on port 8000 (bypass proxy)
    server_url = "http://127.0.0.1:8000/sse"
    headers = {
        "Content-Type": "application/json"
        }
    print(f"Calling extract_apparel_design with {len(image_paths)} image_paths")
    async with sse_client(url=server_url, headers=headers) as (read, write):
        print("Connected to MCP server")
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                # List tools to verify the remote connection
                response = await session.list_tools()
                print(f"Response from list_tools: {response}")
            except Exception as e:
                print(f"Error listing tools: {e}")
                return None
            # FastMCP list_tools() returns a ListToolsResult object containing a 'tools' attribute
            # which is a list of Tool objects.
            tools = response.tools if hasattr(response, 'tools') else response
            print(f"Connected to remote MCP! Tools: {[t.name if hasattr(t, 'name') else t for t in tools]}")

            # Call your tool as usual
            print(f"Calling extract_apparel_design with {len(image_paths)} image_paths")
            result = await session.call_tool("extract_apparel_design", arguments={
                "image_paths": image_paths,
                "product_id": product_id
            })
            return result.content[0].text

async def generate_fashion_prompt(product_id, view, design_json):
    # Connect directly to MCP server on port 8000 (bypass proxy)
    server_url = "http://127.0.0.1:8000/sse"
    headers = {
        "Content-Type": "application/json"
        }
    async with sse_client(url=server_url, headers=headers) as (read, write):
        print("Connected to MCP server")
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                # List tools to verify the remote connection
                response = await session.list_tools()
                print(f"Response from list_tools: {response}")
            except Exception as e:
                print(f"Error listing tools: {e}")
                return None
            # FastMCP list_tools() returns a ListToolsResult object containing a 'tools' attribute
            # which is a list of Tool objects.
            tools = response.tools if hasattr(response, 'tools') else response
            print(f"Connected to remote MCP! Tools: {[t.name if hasattr(t, 'name') else t for t in tools]}")

            # Call your tool as usual
            print(f"Calling generate_prompt_for_view for {product_id} {view}")
            result = await session.call_tool("generate_prompt_for_view", arguments={
                "product_id": product_id,
                "view": view,
                "market_place": "Meesho",
                "design_json": design_json
            })
            print(f"----Result from generate_prompt_for_view: {result.content[0].text}")
            return result.content[0].text


async def main():
    result = await extract_apparel_design(["input_images/front.png", "input_images/back.png"], "123")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
