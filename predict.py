import os
import sys
import json
import logging
from collections import Counter
from pathlib import Path
import numpy as np

from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.logging import RichHandler
from rich.text import Text

import torch
import sentencepiece as spm

from src.transformer import Transformer
from src.parameters import src_vocab_size, trg_vocab_size, pad_id
from src.utils import pad_or_truncate, process_raw_spectrum, logR
from src.predict import greedy_search

# Initialize Rich console
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Comprehensive atom-type descriptions
ATOM_DESCRIPTIONS = {
    # Carbon types
    "[CH3]": "Methyl group (-CH3)",
    "[CH2]": "Methylene group (-CH2-)",
    "[CH]": "Methine group (>CH-)",
    "[C]": "Quaternary carbon (>C<)",
    "[CH2;R]": "Methylene in ring",
    "[CH;R]": "Methine in ring", 
    "[C;R]": "Quaternary carbon in ring",
    
    # Aromatic carbons
    "[c]": "Aromatic carbon",
    "[cH]": "Aromatic CH",
    
    # Oxygen types
    "[OH]": "Hydroxyl group (-OH)",
    "[O]": "Oxygen (ether/carbonyl)",
    "[O;R]": "Oxygen in ring",
    "[o;R]": "Aromatic oxygen in ring",
    
    # Nitrogen types
    "[NH2]": "Primary amine (-NH2)",
    "[NH]": "Secondary amine (>NH)",
    "[N]": "Tertiary amine (>N-)",
    "[NH2;R]": "Primary amine in ring",
    "[NH;R]": "Secondary amine in ring",
    "[N;R]": "Tertiary amine in ring",
    "[n]": "Aromatic nitrogen (pyridine-like)",
    "[nH]": "Aromatic NH (pyrrole-like)",
    
    # Sulfur types
    "[S]": "Sulfur (thioether/disulfide)",
    "[SH]": "Thiol group (-SH)",
    "[SH2]": "Hydrogen sulfide (H2S)",
    "[SH3]": "Sulfonium (SH3+)",
    "[S;R]": "Sulfur in ring",
    "[s;R]": "Aromatic sulfur in ring",
    "[SH;R]": "Thiol in ring",
    "[sH;R]": "Aromatic SH in ring",
    
    # Phosphorus types
    "[P]": "Phosphorus (phosphine/phosphate)",
    "[PH]": "Phosphine (PH)",
    "[PH2]": "Primary phosphine (PH2)",
    "[PH3]": "Phosphine (PH3)",
    "[P;R]": "Phosphorus in ring",
    "[PH;R]": "Phosphine in ring",
    
    # Halogens
    "[F]": "Fluorine",
    "[Cl]": "Chlorine",
    "[Br]": "Bromine",
    "[I]": "Iodine",
    
    # Boron types
    "[B]": "Boron",
    "[BH]": "Borane (BH)",
    "[B;R]": "Boron in ring",
    "[BH;R]": "Borane in ring",
}

def calculate_confidence_score(count, max_count):
    """
    Calculate confidence score as normalized count.
    
    Args:
        count: Number of peaks supporting this rAE
        max_count: Maximum count across all rAEs
    
    Returns:
        Confidence score between 0 and 1
    """
    if max_count == 0:
        return 0.0
    
    # Simple normalization: count / max_count
    confidence = count / max_count
    return round(confidence, 3)

def get_confidence_level(score):
    """Convert confidence score to categorical level with color."""
    if score >= 0.7:
        return "HIGH", "green"
    elif score >= 0.09:
        return "MEDIUM", "yellow"
    else:
        return "LOW", "red"

def format_human_output(results):
    """Create human-readable console output."""
    
    console.print("\n" + "═"*65)
    console.print(" "*20 + "MASSSPEC-REFAINER RESULTS")
    console.print("═"*65 + "\n")
    
    # Summary
    console.print("Spectrum Analysis Summary:")
    console.print(f"   Input peaks: {len(results['src'].split())}")
    console.print(f"   Processed peaks: {len(results['src'].split())}\n")
    
    # Print all rAEs list (filter empty strings for display)
    all_raes_clean = sorted([rae for rae in results['pred_all_rAEs'] if rae and rae.strip()])
    console.print("All Predicted Atom-Types (rAE_all):")
    console.print(f"   {all_raes_clean}\n")
    
    # Print molecular rAEs list (filter empty strings for display)
    mol_raes_clean = sorted([rae for rae in results['pred_mol_rAEs'] if rae and rae.strip()])
    console.print("Molecular Atom-Types (rAE_mol):")
    console.print(f"   {mol_raes_clean}\n")
    
    # All predictions table (not just molecular)
    console.print("Predicted Atom-Types (All rAEs):")
    console.print(f"   Total unique atom-types: {len(all_raes_clean)}\n")
    
    # Create confidence scores
    all_counts = results['pred_all_rAEs_count']
    max_count = max(all_counts.values()) if all_counts else 1
    
    # Create table for predictions
    table = Table(show_header=True)
    table.add_column("Atom-Type", width=12)
    # table.add_column(" ", justify="left", width=15)
    table.add_column("Score", justify="center", width=20)
    table.add_column("Count", justify="right", width=5)
    table.add_column("Description", width=35)
    
    all_predictions = []
    
    # Sort by confidence (count) for better presentation
    sorted_raes = sorted(results['pred_all_rAEs'], 
                        key=lambda x: all_counts.get(x, 0), 
                        reverse=True)
    
    for atom_type in sorted_raes:
        # Skip empty atom types
        if not atom_type or atom_type.strip() == "":
            continue
            
        count = all_counts.get(atom_type, 0)
        confidence = calculate_confidence_score(count, max_count)
        level, _ = get_confidence_level(confidence)
        
        # Create confidence bar (no color)
        bar_length = int(confidence * 10)
        conf_bar = f"{'█' * bar_length}{'░' * (10 - bar_length)}"
        
        # Combine score and bar
        # bar = f"{conf_bar}"
        # score = f"{confidence:.1%}"
        score_with_bar = f"{conf_bar:<11} {confidence:>6.1%}"

        description = ATOM_DESCRIPTIONS.get(atom_type, "Unknown group")
        
        table.add_row(
            atom_type,
            score_with_bar,
            str(count),
            description
        )
        
        all_predictions.append({
            "atom_type": atom_type,
            "confidence": confidence,
            "level": level,
            "count": count,
            "is_molecular": atom_type in results['pred_mol_rAEs']
        })
    
    console.print(table)
    
    # Refinement suggestions based on count criteria
    # Include if count > 1
    consider_including = [p["atom_type"] for p in all_predictions if p["count"] > 1]
    # Verify if fragment-only with count = 1
    verify_presence = [p["atom_type"] for p in all_predictions 
                      if p["count"] == 1 and not p["is_molecular"]]
    
    console.print("\nLibrary Search Refinement Suggestions:")
    if consider_including:
        console.print(f"   Consider INCLUDING compounds with: {', '.join(consider_including)}")
    if verify_presence:
        console.print(f"   Verify presence of: {', '.join(verify_presence)}")
    
    # Overall confidence
    avg_confidence = np.mean([p["confidence"] for p in all_predictions]) if all_predictions else 0
    console.print(f"\nOverall Prediction Confidence: {avg_confidence:.1%}\n")
    
    return all_predictions, avg_confidence

def save_enhanced_json(results, all_predictions, avg_confidence, output_path):
    """Save enhanced JSON output with confidence scores and descriptions."""
    
    all_counts = results['pred_all_rAEs_count']
    max_count = max(all_counts.values()) if all_counts else 1
    
    enhanced_results = {
        "summary": {
            "input_peaks": len(results['src'].split()),
            "total_predicted_atoms": len(results['pred_all_rAEs']),
            "unique_molecular_raes": len(results['pred_mol_rAEs']),
            "overall_confidence": round(avg_confidence, 3)
        },
        "molecular_predictions": [],
        "all_fragment_predictions": {},
        "refinement_suggestions": {
            "high_confidence_inclusions": [],
            "moderate_confidence": [],
            "low_confidence_uncertain": []
        },
        "raw_data": results
    }
    
    # Add molecular predictions with confidence
    for atom_type in results['pred_mol_rAEs']:
        count = all_counts.get(atom_type, 0)
        confidence = calculate_confidence_score(count, max_count)
        level, _ = get_confidence_level(confidence)
        
        pred_entry = {
            "atom_type": atom_type,
            "description": ATOM_DESCRIPTIONS.get(atom_type, "Unknown"),
            "peak_count": count,
            "confidence_score": confidence,
            "confidence_level": level
        }
        
        enhanced_results["molecular_predictions"].append(pred_entry)
        
        # Categorize for refinement suggestions
        if level == "HIGH":
            enhanced_results["refinement_suggestions"]["high_confidence_inclusions"].append(atom_type)
        elif level == "MEDIUM":
            enhanced_results["refinement_suggestions"]["moderate_confidence"].append(atom_type)
        else:
            enhanced_results["refinement_suggestions"]["low_confidence_uncertain"].append(atom_type)
    
    # Add all fragment predictions
    for atom_type, count in all_counts.items():
        confidence = calculate_confidence_score(count, max_count)
        enhanced_results["all_fragment_predictions"][atom_type] = {
            "count": count,
            "confidence": confidence,
            "description": ATOM_DESCRIPTIONS.get(atom_type, "Unknown")
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    return enhanced_results

def model_generator(model, tokenized_input, device):
    """Run model inference."""
    src_data = torch.LongTensor(pad_or_truncate(tokenized_input)).unsqueeze(0).to(device)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        src_data = model.src_embedding(src_data)
        src_data = model.positional_encoder(src_data)
        e_output = model.encoder(src_data, e_mask)
        result = greedy_search(model, e_output, e_mask)
    
    return result

def main(args):
    # Show loading message
    with console.status("Loading model...") as status:
        # Model initialization
        model = Transformer(src_vocab_size, trg_vocab_size)
        state_dicts = torch.load(args.checkpoint_path, map_location=args.device)
        if 'model_state_dict' in state_dicts:
            model.load_state_dict(state_dicts['model_state_dict'])
        else:
            model.load_state_dict(state_dicts)
        model.to(args.device)
        
        # Tokenizer initialization
        src_sp = spm.SentencePieceProcessor(model_file="data/sp/src_sp.model")
        trg_sp = spm.SentencePieceProcessor(model_file="data/sp/trg_sp.model")
    
    console.print("Model loaded successfully!\n")
    
    # Prepare input data
    input_src_list = []
    if args.preprocess:
        console.print("Preprocessing spectrum...")
        input_text = process_raw_spectrum(
            args.spectrum_file,
            normalize=False,
        )
        input_text[:, 1] = logR(input_text[:, 1])
        input_text = input_text.astype(int)
        input_text = ' '.join([f'{mz}i{intensity}' for mz, intensity in input_text])
        input_src_list.append(input_text)
    else:
        for datapoint in open(args.spectrum_file, 'r'):
            input_src_list.append(datapoint.strip())
    
    tokenized_list = [pad_or_truncate(src_sp.EncodeAsIds(input_text)) for input_text in input_src_list]

    # Inference
    results = []
    for i, tokenized_input in enumerate(track(tokenized_list, description="Processing spectra...")):
        decoded_ids = model_generator(model, tokenized_input, args.device)
        result = trg_sp.decode_ids(decoded_ids)
        
        # Process all rAEs - filter out empty strings
        all_rAEs_raw = result.strip().replace('<mol>', '').replace('.', ' ').split()
        all_rAEs_raw = [rae.strip() for rae in all_rAEs_raw if rae != '']  # Filter empty strings

        all_rAEs = set(rae for rae in all_rAEs_raw if rae)  # Filter empty strings
        all_rAEs_count = Counter(rae for rae in all_rAEs_raw if rae)  # Filter empty strings
        
        # Process molecular rAEs - filter out empty strings
        if " <mol> " in result:
            mol_rAEs_raw = result.split(" <mol> ")[-1].strip().split(" ")
            mol_rAEs = set(rae.strip() for rae in mol_rAEs_raw if rae)  # Filter empty strings
        else:
            mol_rAEs = set()
        
        # Empty strings are already filtered out above
        
        result_dict = {
            "src": input_src_list[i],
            "raw_output": result,
            "pred_all_rAEs": list(all_rAEs),
            "pred_all_rAEs_count": dict(all_rAEs_count),
            "pred_mol_rAEs": list(mol_rAEs),
        }
        
        results.append(result_dict)
        
        # Display human-readable output
        all_predictions, avg_confidence = format_human_output(result_dict)
        
        # Save enhanced JSON
        enhanced_results = save_enhanced_json(result_dict, all_predictions, avg_confidence, args.output)
    
    console.print(f"Results saved to {args.output}")
    
    if args.verbose:
        console.print("\nRaw output saved for debugging purposes.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MassSpec-RefAIner: Predict atomic environments from EI-MS spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python predict.py --spectrum_file spectrum.txt --output results.json
  
  With preprocessing:
    python predict.py --spectrum_file raw_spectrum.txt --preprocess --output results.json
  
  Verbose mode:
    python predict.py --spectrum_file spectrum.txt --verbose --output results.json
        """
    )
    
    parser.add_argument(
        '--spectrum_file', type=Path, required=True,
        help="Path to spectrum file (tab-delimited m/z and intensity values)"
    )
    parser.add_argument(
        '-pre', '--preprocess', action='store_true',
        help="Preprocess raw spectrum (normalize, filter noise)"
    )
    parser.add_argument(
        '-o', '--output', type=Path, default='results.json',
        help="Output file path (default: results.json)"
    )
    parser.add_argument(
        '--checkpoint_path', type=Path, default='saved_models/checkpoint.pth',
        help="Model checkpoint path"
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Computation device (cuda/cpu)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Show detailed output including fragment predictions"
    )
    
    args = parser.parse_args()
    
    if not args.spectrum_file.exists():
        console.print(f"Error: Spectrum file '{args.spectrum_file}' not found!")
        sys.exit(1)
    
    main(args)
