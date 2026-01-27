from .app import initialize_tools
from .server import run


if __name__ == "__main__":
    # Initialize tools first (this triggers all @mcp.tool() decorators)
    initialize_tools()
    
    run()
