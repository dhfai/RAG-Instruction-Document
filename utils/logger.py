import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
import colorlog

from config.settings import settings

# Install rich traceback handler
install(show_locals=True)

# Console for rich output
console = Console()

class BeautifulLogger:
    """Beautiful logging system with colors and rich formatting"""

    def __init__(self):
        self._setup_loguru()
        self._setup_colorlog()

    def _setup_loguru(self):
        """Setup loguru with beautiful formatting"""
        # Remove default logger
        loguru_logger.remove()

        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Console logger with colors
        loguru_logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True
        )

        # File logger
        loguru_logger.add(
            log_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 month",
            compression="zip"
        )

        # Error file logger
        loguru_logger.add(
            log_dir / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="5 MB",
            retention="3 months"
        )

    def _setup_colorlog(self):
        """Setup colorlog for additional formatters"""
        self.color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

# Global logger instance
beautiful_logger = BeautifulLogger()

def get_logger(name: str) -> loguru_logger.__class__:
    """Get a logger instance with the given name"""
    return loguru_logger.bind(name=name)

# Convenience functions
def log_info(message: str, component: str = "SYSTEM"):
    """Log info message with component context"""
    logger = get_logger(component)
    logger.info(message)

def log_error(message: str, component: str = "SYSTEM", exception: Optional[Exception] = None):
    """Log error message with optional exception"""
    logger = get_logger(component)
    if exception:
        logger.exception(f"{message}: {exception}")
    else:
        logger.error(message)

def log_warning(message: str, component: str = "SYSTEM"):
    """Log warning message"""
    logger = get_logger(component)
    logger.warning(message)

def log_debug(message: str, component: str = "SYSTEM"):
    """Log debug message"""
    logger = get_logger(component)
    logger.debug(message)

def log_success(message: str, component: str = "SYSTEM"):
    """Log success message"""
    logger = get_logger(component)
    logger.success(message)

# Rich console functions for special output
def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🎓 ORCHESTRATED RAG SYSTEM - MODUL AJAR GENERATOR 🎓                    ║
║                                                                              ║
║    Sistem RAG Multi-Strategi untuk Pembuatan Modul Ajar Otomatis           ║
║    dengan Optimisasi Matematis dan Validasi Komprehensif                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")

def print_section_header(title: str):
    """Print section header"""
    console.print(f"\n📋 {title}", style="bold yellow")
    console.print("─" * (len(title) + 4), style="yellow")

def print_progress(message: str, step: int = None, total: int = None):
    """Print progress message"""
    if step and total:
        progress = f"[{step}/{total}]"
        console.print(f"⏳ {progress} {message}", style="blue")
    else:
        console.print(f"⏳ {message}", style="blue")

def print_success(message: str):
    """Print success message"""
    console.print(f"✅ {message}", style="bold green")

def print_error(message: str):
    """Print error message"""
    console.print(f"❌ {message}", style="bold red")

def print_warning(message: str):
    """Print warning message"""
    console.print(f"⚠️  {message}", style="bold yellow")

def print_info(message: str):
    """Print info message"""
    console.print(f"ℹ️  {message}", style="cyan")

def print_strategy_selection(retrieval: str, generation: str, confidence: float):
    """Print strategy selection info"""
    console.print(f"\n🎯 Strategy Selection:", style="bold magenta")
    console.print(f"   Retrieval: {retrieval}", style="green")
    console.print(f"   Generation: {generation}", style="green")
    console.print(f"   Confidence: {confidence:.2%}", style="yellow")

def print_validation_result(result):
    """Print validation result"""
    console.print(f"\n🔍 Validation Results:", style="bold cyan")
    console.print(f"   Faithfulness: {result.faithfulness_score:.2%}", style="green")
    console.print(f"   Numeric Consistency: {'✅' if result.numeric_consistency else '❌'}")
    console.print(f"   Contradiction Detected: {'❌' if result.contradiction_detected else '✅'}")
    console.print(f"   Overall Confidence: {result.overall_confidence:.2%}", style="yellow")

def print_processing_stats(stats: dict):
    """Print processing statistics"""
    console.print(f"\n📊 Processing Statistics:", style="bold blue")
    for key, value in stats.items():
        if isinstance(value, float):
            console.print(f"   {key}: {value:.2f}", style="white")
        else:
            console.print(f"   {key}: {value}", style="white")

def print_table(data: list, headers: list, title: str = None):
    """Print data in table format"""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")

    # Add columns
    for header in headers:
        table.add_column(header, style="cyan")

    # Add rows
    for row in data:
        table.add_row(*[str(cell) for cell in row])

    if title:
        console.print(f"\n{title}", style="bold yellow")
    console.print(table)

def log_system_startup():
    """Log system startup information"""
    log_info("🚀 Orchestrated RAG System Starting Up", "STARTUP")
    log_info(f"📁 Data Path: {settings.DATA_PATH}", "STARTUP")
    log_info(f"🗄️ MongoDB URL: {settings.MONGODB_URL}", "STARTUP")
    log_info(f"🧠 Embedding Model: {settings.EMBEDDINGS_MODEL}", "STARTUP")
    log_info(f"🤖 Default LLM: {settings.DEFAULT_LLM_MODEL}", "STARTUP")

def log_component_status(component: str, status: str, details: str = ""):
    """Log component status"""
    status_emoji = {
        "STARTING": "🔄",
        "READY": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️"
    }

    emoji = status_emoji.get(status, "ℹ️")
    message = f"{emoji} {component}: {status}"
    if details:
        message += f" - {details}"

    if status == "ERROR":
        log_error(message, component)
    elif status == "WARNING":
        log_warning(message, component)
    else:
        log_info(message, component)
