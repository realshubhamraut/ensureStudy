"""
Cross-platform beep sound module.
Uses winsound on Windows, and os.system with afplay/paplay on macOS/Linux.
"""
import platform
import os

def beep(frequency=2500, duration=1000):
    """
    Play a beep sound.
    
    Args:
        frequency: Frequency in Hz (used on Windows)
        duration: Duration in milliseconds
    """
    system = platform.system()
    
    if system == "Windows":
        import winsound
        winsound.Beep(frequency, duration)
    elif system == "Darwin":  # macOS
        # Use afplay with a system sound or generate a beep
        os.system('afplay /System/Library/Sounds/Ping.aiff &')
    else:  # Linux
        # Try using paplay or beep command
        os.system('paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || echo -e "\\a"')
