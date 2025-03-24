import argparse
import os
import sys
import time



import scGPT_dataloader
import cellxgene_dataloader
import scGPT_embedder
import scGPT_classifier

config_file = "scripts\models\scGPT_Human_config.json"


def main():
    parser = argparse.ArgumentParser(description='scGPT pipeline')
    parser.add_argument('--url', type=str, help='URL of the cellxgene dataset')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--file_name', type=str, help="File to be analyzed")
    parser.add_argument('--model', default = "scripts\models\scGPT_Human", help='Path to the scGPT model used for embedding. Default is scGPT_Human')
    args = parser.parse_args()

    dataloader = scGPT_dataloader.scGPT_dataloader(args.file_name, args.output_dir)







