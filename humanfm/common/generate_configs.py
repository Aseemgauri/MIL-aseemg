#!/usr/bin/env python3
"""
Configuration Generator for HumanFM System
==========================================

This script reads the central system.conf file and generates optimized,
static configuration files for each component in their respective directories.

Generated files:
- GST_Client/gst_config.h - C++ header with #define macros
- SemanticListening/SL-RT/src/sl_config.h - C++ header with #define macros
- python_server/config.py - Python module with constants
- node_server/config.js - JavaScript module with constants

Usage:
    python3 generate_configs.py

This eliminates runtime config lookup overhead while maintaining
a single source of truth in system.conf.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union


class ConfigGenerator:
    def __init__(self, config_file: str = "common/system.conf"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from system.conf file"""
        try:
            with open(self.config_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert values to appropriate types
                        self.config[key] = self.parse_value(value)
                    else:
                        print(f"Warning: Invalid line {line_num}: {line}")
        
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading config file: {e}")
            sys.exit(1)
    
    def parse_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse string value into appropriate type"""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # String values (remove quotes if present)
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        
        return value
    
    def generate_cpp_header(self, filename: str, defines: Dict[str, str], 
                          header_guard: str, description: str):
        """Generate C++ header file with #define macros"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(f"#ifndef {header_guard}\n")
            f.write(f"#define {header_guard}\n\n")
            f.write("/*\n")
            f.write(f" * {description}\n")
            f.write(" * \n")
            f.write(" * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY!\n")
            f.write(" * Generated from system.conf by generate_configs.py\n")
            f.write(" * \n")
            f.write(" * To change configuration values, edit system.conf and run:\n")
            f.write(" * python3 generate_configs.py\n")
            f.write(" */\n\n")
            
            for key, value in defines.items():
                f.write(f"#define {key} {value}\n")
            
            f.write(f"\n#endif // {header_guard}\n")
    
    def generate_python_module(self, filename: str, constants: Dict[str, Any], 
                             description: str):
        """Generate Python module with constants"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write('"""\n')
            f.write(f"{description}\n")
            f.write("\n")
            f.write("AUTO-GENERATED FILE - DO NOT EDIT MANUALLY!\n")
            f.write("Generated from system.conf by generate_configs.py\n")
            f.write("\n")
            f.write("To change configuration values, edit system.conf and run:\n")
            f.write("python3 generate_configs.py\n")
            f.write('"""\n\n')
            
            for key, value in constants.items():
                if isinstance(value, str):
                    f.write(f"{key} = '{value}'\n")
                elif isinstance(value, bool):
                    f.write(f"{key} = {str(value)}\n")
                else:
                    f.write(f"{key} = {value}\n")
    
    def generate_javascript_module(self, filename: str, constants: Dict[str, Any], 
                                 description: str):
        """Generate JavaScript module with constants"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("/*\n")
            f.write(f" * {description}\n")
            f.write(" * \n")
            f.write(" * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY!\n")
            f.write(" * Generated from system.conf by generate_configs.py\n")
            f.write(" * \n")
            f.write(" * To change configuration values, edit system.conf and run:\n")
            f.write(" * python3 generate_configs.py\n")
            f.write(" */\n\n")
            
            for key, value in constants.items():
                if isinstance(value, str):
                    f.write(f"const {key} = '{value}';\n")
                elif isinstance(value, bool):
                    f.write(f"const {key} = {str(value).lower()};\n")
                else:
                    f.write(f"const {key} = {value};\n")
            
            f.write("\nmodule.exports = {\n")
            for key in constants.keys():
                f.write(f"  {key},\n")
            f.write("};\n")
    
    def cpp_value(self, value: Any) -> str:
        """Convert Python value to C++ #define value"""
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)
    
    def generate_gst_config(self):
        """Generate GST Client configuration header"""
        defines = {
            # Global settings
            "SYSTEM_SAMPLE_RATE": self.cpp_value(self.config["SYSTEM_SAMPLE_RATE"]),
            "SYSTEM_NUM_CLASSES": self.cpp_value(self.config["SYSTEM_NUM_CLASSES"]),
            "SYSTEM_STREAMING_MODE": self.cpp_value(self.config["SYSTEM_STREAMING_MODE"]),
            "SYSTEM_ORANGEPI_IP": self.cpp_value(self.config["SYSTEM_ORANGEPI_IP"]),
            "SYSTEM_LAPTOP_IP": self.cpp_value(self.config["SYSTEM_LAPTOP_IP"]),
            "SYSTEM_UDP_STREAMING_PORT": self.cpp_value(self.config["SYSTEM_UDP_STREAMING_PORT"]),
            "SYSTEM_MULTICAST_ADDRESS": self.cpp_value(self.config["SYSTEM_MULTICAST_ADDRESS"]),
            
            # GST specific settings
            "GST_AUDIO_MODE": self.cpp_value(self.config["GST_AUDIO_MODE"]),
            "GST_AUDIO_FILE_DIRECTORY": self.cpp_value(self.config["GST_AUDIO_FILE_DIRECTORY"]),
            "GST_DEFAULT_AUDIO_FILE": self.cpp_value(self.config["GST_DEFAULT_AUDIO_FILE"]),
            "GST_VOLUME_COMPENSATION": self.cpp_value(self.config["GST_VOLUME_COMPENSATION"]),
            "GST_AMPLIFY_COMPENSATION": self.cpp_value(self.config["GST_AMPLIFY_COMPENSATION"]),
            "GST_AUDIO_BACKEND": self.cpp_value(self.config["GST_AUDIO_BACKEND"]),
            "GST_ALSA_DEVICE": self.cpp_value(self.config["GST_ALSA_DEVICE"]),
            "GST_JACK_CLIENT_NAME": self.cpp_value(self.config["GST_JACK_CLIENT_NAME"]),
            "GST_BROADCAST_ADDRESS": self.cpp_value(self.config["GST_BROADCAST_ADDRESS"]),
            
            # Derived values
            "GST_USE_FILE_AUDIO": "1" if self.config["GST_AUDIO_MODE"] == "file" else "0",
        }
        
        self.generate_cpp_header(
            "GST_Client/gst_config.h",
            defines,
            "GST_CONFIG_H",
            "GST Client Configuration"
        )
    
    def generate_sl_config(self):
        """Generate Semantic Listening Client configuration header"""
        defines = {
            # Global settings
            "SYSTEM_SAMPLE_RATE": self.cpp_value(self.config["SYSTEM_SAMPLE_RATE"]),
            "SYSTEM_NUM_CLASSES": self.cpp_value(self.config["SYSTEM_NUM_CLASSES"]),
            
            # SL specific settings
            "SL_JACK_CLIENT_NAME": self.cpp_value(self.config["SL_JACK_CLIENT_NAME"]),
            "SL_GAIN_COMPENSATION_FACTOR": self.cpp_value(self.config["SL_GAIN_COMPENSATION_FACTOR"]),
            "SL_MAX_JACK_BUFFER_SIZE": self.cpp_value(self.config["SL_MAX_JACK_BUFFER_SIZE"]),
            "SL_INPUT_PORTS": self.cpp_value(self.config["SL_INPUT_PORTS"]),
            "SL_OUTPUT_PORTS": self.cpp_value(self.config["SL_OUTPUT_PORTS"]),
            "SL_SOCKET_PATH": self.cpp_value(self.config["SOCKET_PATH"]),
            
            # Feature flags
            "SL_ENABLE_PASSTHROUGH": "1" if self.config["SL_ENABLE_PASSTHROUGH"] else "0",
            "SL_ENABLE_INFERENCE": "1" if self.config["SL_ENABLE_INFERENCE"] else "0", 
            "SL_FILL_EMBEDDING": "1" if self.config["SL_FILL_EMBEDDING"] else "0",
        }
        
        self.generate_cpp_header(
            "SemanticListening/SL-RT/src/sl_config.h",
            defines,
            "SL_CONFIG_H", 
            "Semantic Listening Client Configuration"
        )
    
    def generate_python_config(self):
        """Generate Python Server configuration module"""
        constants = {
            # Global settings
            "SYSTEM_SAMPLE_RATE": self.config["SYSTEM_SAMPLE_RATE"],
            "SYSTEM_NUM_CLASSES": self.config["SYSTEM_NUM_CLASSES"],
            "SYSTEM_CLASS_NAMES": self.config["SYSTEM_CLASS_NAMES"].split(','),
            "SYSTEM_STREAMING_MODE": self.config["SYSTEM_STREAMING_MODE"],
            "SYSTEM_UDP_STREAMING_PORT": self.config["SYSTEM_UDP_STREAMING_PORT"],
            "SYSTEM_MULTICAST_ADDRESS": self.config["SYSTEM_MULTICAST_ADDRESS"],
            "SYSTEM_NODE_SERVER_PORT": self.config["SYSTEM_NODE_SERVER_PORT"],
            
            # Python specific settings
            "PYTHON_DEMO_MODE": self.config["PYTHON_DEMO_MODE"],
            "PYTHON_DEMO_CLASS": self.config["PYTHON_DEMO_CLASS"],
            "PYTHON_CLASS_THRESHOLD": self.config["PYTHON_CLASS_THRESHOLD"],
            "PYTHON_CONFIDENCE_THRESHOLD": self.config["PYTHON_CONFIDENCE_THRESHOLD"],
            "PYTHON_BUFFER_SIZE": self.config["PYTHON_BUFFER_SIZE"],
            "PYTHON_MAX_CONNECTIONS": self.config["PYTHON_MAX_CONNECTIONS"],
        }
        
        self.generate_python_module(
            "WindowsPythonServer/config.py",
            constants,
            "Python Server Configuration"
        )
    
    def generate_node_config(self):
        """Generate Node Server configuration module"""
        constants = {
            # Global settings
            "SYSTEM_SAMPLE_RATE": self.config["SYSTEM_SAMPLE_RATE"],
            "SYSTEM_NUM_CLASSES": self.config["SYSTEM_NUM_CLASSES"],
            "SYSTEM_CLASS_NAMES": self.config["SYSTEM_CLASS_NAMES"].split(','),
            "SYSTEM_NODE_SERVER_PORT": self.config["SYSTEM_NODE_SERVER_PORT"],
            "SYSTEM_LAPTOP_IP": self.config["SYSTEM_LAPTOP_IP"],
            
            # Node specific settings
            "NODE_DATA_EXPIRY_TIME_SECONDS": self.config["NODE_DATA_EXPIRY_TIME_SECONDS"],
            "NODE_MINIMUM_ACTIVATION_TIME_MS": self.config["NODE_MINIMUM_ACTIVATION_TIME_MS"],
            "NODE_MAX_EVENTS_PER_CLASS": self.config["NODE_MAX_EVENTS_PER_CLASS"],
            "NODE_MAX_TOTAL_EVENTS": self.config["NODE_MAX_TOTAL_EVENTS"],
            "NODE_SOCKET_PATH": self.config["SOCKET_PATH"],
            "NODE_DATABASE_FILE": self.config["NODE_DATABASE_FILE"],
            "NODE_DEFAULT_CLASS_LEVEL": self.config["NODE_DEFAULT_CLASS_LEVEL"],
        }
        
        self.generate_javascript_module(
            "NodeServer/config.js",
            constants,
            "Node Server Configuration"
        )
    
    def generate_all(self):
        """Generate all configuration files"""
        print("🔧 Generating configuration files from system.conf...")
        
        try:
            self.generate_gst_config()
            print("✅ Generated GST_Client/gst_config.h")
            
            self.generate_sl_config()
            print("✅ Generated SemanticListening/SL-RT/src/sl_config.h")
            
            self.generate_python_config()
            print("✅ Generated WindowsPythonServer/config.py")
            
            self.generate_node_config()
            print("✅ Generated NodeServer/config.js")
            
            print("🎯 All configuration files generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating config files: {e}")
            sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        return
    
    generator = ConfigGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main() 