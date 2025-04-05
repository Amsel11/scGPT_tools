import argparse
import json
import os
from datetime import datetime

def create_argparser():
    parser = argparse.ArgumentParser(description='Config Test')
    parser.add_argument('--input_file', default="test_data.csv", help='Input file path')
    parser.add_argument('--config_file', default=None, help='Config file path')
    parser.add_argument('--param1', type=int, default=10, help='Test parameter 1')
    parser.add_argument('--param2', type=str, default="default", help='Test parameter 2')
    parser.add_argument('--stage', choices=['analyze', 'process', 'finalize'], help='Processing stage')
    return parser

def load_config(config_path):
    print(f"Loading config from {config_path}")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def build_config(args):
    # Start with defaults from args
    config = vars(args).copy()
    
    # Load from config file if provided
    if args.config_file:
        file_config = load_config(args.config_file)
        
        # Update config with file values (except those explicitly set in CLI)
        for key, value in file_config.items():
            if key in config and key not in ['input_file', 'config_file', 'stage'] and value is not None:
                config[key] = value
                print(f"Using config file value for {key}: {value}")
    
    # Display final config
    print("\nFinal configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    
    return config

def simulate_analysis(config):
    print("\n=== Running Analysis ===")
    
    # Simulate finding parameters from analysis
    auto_config = {
        "param1": config["param1"] * 2,  # Double the param1 value
        "param2": f"analyzed_{config['param2']}",  # Update param2
        "detected_param": "auto_detected_value",  # Add a new parameter
        "timestamp": datetime.now().isoformat()
    }
    
    # Save the auto config
    output_file = "auto_config.json"
    with open(output_file, 'w') as f:
        json.dump(auto_config, f, indent=2)
    
    print(f"Analysis complete. Auto-config saved to {output_file}")
    return auto_config

def simulate_processing(config):
    print("\n=== Running Processing ===")
    
    # Simulate updating config during processing
    config["processed"] = True
    config["processing_timestamp"] = datetime.now().isoformat()
    
    # Save the updated config
    output_file = "processing_config.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Processing complete. Updated config saved to {output_file}")
    return config

def simulate_finalization(config):
    print("\n=== Running Finalization ===")
    
    # Simulate final updates
    config["finalized"] = True
    config["finalization_timestamp"] = datetime.now().isoformat()
    
    # Save the final config
    output_file = "final_config.json"
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Finalization complete. Final config saved to {output_file}")
    return config

def main():
    parser = create_argparser()
    args = parser.parse_args()
    
    # Build initial config
    config = build_config(args)
    
    # Run the requested stage
    if args.stage == 'analyze':
        auto_config = simulate_analysis(config)
        config.update(auto_config)
    elif args.stage == 'process':
        config = simulate_processing(config)
    elif args.stage == 'finalize':
        config = simulate_finalization(config)
    else:
        print("No stage specified. Use --stage [analyze|process|finalize]")

if __name__ == "__main__":
    main()