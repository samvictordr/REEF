#!/usr/bin/env python3
"""
Excel Workbook Extraction Script for Coral Multi-Omics Data.

This script inspects the Supplemental_Data.xlsx workbook, detects sheet headers,
proposes column mappings, and extracts canonical TSV matrices for downstream analysis.

Dataset: Williams et al. (2021) - Multi-omic characterization of the thermal 
stress phenome in the stony coral Montipora capitata
DOI: 10.7717/peerj.12335

Usage:
    # Inspect headers and create schema report
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --report
    
    # Auto-extract using suggested mappings
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --auto --out data/processed/
    
    # Extract with confirmed mapping file
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --mapping data/column_map_confirmed.yaml
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Keywords for detecting column types
SAMPLE_ID_KEYWORDS = ['sample', 'id', 'colony', 'replicate', 'name', 'specimen']
# 'accession' should be first/prioritized as the primary feature identifier
FEATURE_ID_KEYWORDS = ['accession', 'gene', 'metabolite', 'asv', 'otu', 'compound', 'peptide']
# Exclude 'protein' from feature ID keywords since 'Protein FDR Confidence' is metadata, not identifier
PROTEIN_METADATA_KEYWORDS = ['protein fdr', 'protein confidence', 'master protein']
METADATA_KEYWORDS = ['treatment', 'time', 'condition', 'temp', 'temperature', 'stress', 'bleach', 'color']
VALUE_KEYWORDS = ['norm', 'abund', 'count', 'read', 'peak', 'intensity', 'expression', 'fold']


class ExcelSheetInspector:
    """Inspects Excel sheets and detects column types."""
    
    def __init__(self, sheet_name: str, df: pd.DataFrame):
        self.sheet_name = sheet_name
        self.df = df
        self.analysis = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Run full analysis on the sheet."""
        self.analysis = {
            'sheet_name': self.sheet_name,
            'shape': list(self.df.shape),
            'columns': list(self.df.columns),
            'dtypes': {str(k): str(v) for k, v in self.df.dtypes.items()},
            'first_rows_preview': self.df.head(3).to_dict('records'),
            'candidate_columns': self._detect_column_candidates(),
            'orientation_guess': self._guess_orientation(),
            'null_counts': self.df.isnull().sum().to_dict(),
        }
        return self.analysis
    
    def _detect_column_candidates(self) -> Dict[str, List[str]]:
        """Detect candidate columns for sample ID, feature ID, and values."""
        candidates = {
            'sample_id': [],
            'feature_id': [],
            'metadata': [],
            'values': [],
        }
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            
            # Skip protein metadata columns (e.g., 'Protein FDR Confidence: Combined')
            if any(kw in col_lower for kw in PROTEIN_METADATA_KEYWORDS):
                candidates['metadata'].append(str(col))
                continue
            
            # Check against keyword lists
            if any(kw in col_lower for kw in SAMPLE_ID_KEYWORDS):
                candidates['sample_id'].append(str(col))
            if any(kw in col_lower for kw in FEATURE_ID_KEYWORDS):
                candidates['feature_id'].append(str(col))
            if any(kw in col_lower for kw in METADATA_KEYWORDS):
                candidates['metadata'].append(str(col))
            if any(kw in col_lower for kw in VALUE_KEYWORDS):
                candidates['values'].append(str(col))
            
            # Check data type for numeric columns (likely value columns)
            if self.df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if str(col) not in candidates['values']:
                    # Only add if column name doesn't look like an ID
                    if not any(kw in col_lower for kw in SAMPLE_ID_KEYWORDS + FEATURE_ID_KEYWORDS):
                        candidates['values'].append(str(col))
        
        return candidates
    
    def _guess_orientation(self) -> Dict[str, Any]:
        """Guess whether rows are samples or features."""
        n_rows, n_cols = self.df.shape
        
        # Heuristics:
        # - If many more columns than rows, likely rows=samples, cols=features
        # - If first column looks like sample IDs, rows=samples
        # - If column names look like sample IDs, cols=samples
        
        first_col = self.df.columns[0]
        first_col_lower = str(first_col).lower()
        
        rows_are_samples_score = 0
        
        # More columns suggests features as columns
        if n_cols > n_rows * 2:
            rows_are_samples_score += 2
        elif n_rows > n_cols * 2:
            rows_are_samples_score -= 2
        
        # First column looks like sample ID
        if any(kw in first_col_lower for kw in SAMPLE_ID_KEYWORDS):
            rows_are_samples_score += 3
        
        # Check if column names look numeric (often feature IDs like gene names)
        numeric_col_names = sum(1 for c in self.df.columns if re.match(r'^\d', str(c)))
        if numeric_col_names > len(self.df.columns) * 0.5:
            rows_are_samples_score += 1
        
        return {
            'rows_are_samples': rows_are_samples_score > 0,
            'confidence': abs(rows_are_samples_score),
            'n_rows': n_rows,
            'n_cols': n_cols,
        }


def inspect_workbook(excel_path: Path) -> Dict[str, Any]:
    """
    Inspect an Excel workbook and analyze all sheets.
    
    Args:
        excel_path: Path to the Excel file
        
    Returns:
        Dictionary with full schema report
    """
    logger.info(f"Inspecting workbook: {excel_path}")
    
    # Read all sheets with header=1 (skip description row in Williams et al. data)
    try:
        sheets = pd.read_excel(excel_path, sheet_name=None, header=1)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        raise
    
    report = {
        'file_path': str(excel_path),
        'inspection_time': datetime.now().isoformat(),
        'n_sheets': len(sheets),
        'sheet_names': list(sheets.keys()),
        'sheets': {},
    }
    
    for sheet_name, df in sheets.items():
        logger.info(f"  Analyzing sheet: {sheet_name} ({df.shape[0]} x {df.shape[1]})")
        inspector = ExcelSheetInspector(sheet_name, df)
        report['sheets'][sheet_name] = inspector.analyze()
    
    return report


def create_column_mapping(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create suggested column mapping from schema report.
    
    Args:
        report: Schema report from inspect_workbook
        
    Returns:
        Suggested column mapping dictionary
    """
    mapping = {
        'created': datetime.now().isoformat(),
        'source_file': report['file_path'],
        'sheets': {},
    }
    
    # Map sheet names to omic types (handle both underscore and space variants)
    sheet_to_omic = {
        'Data S1': 'proteomics',
        'Data_S1': 'proteomics',
        'Data S2': 'metabolomics',
        'Data_S2': 'metabolomics', 
        'Data S3': 'transcriptomics',
        'Data_S3': 'transcriptomics',
        'Data S4': 'microbiome',
        'Data_S4': 'microbiome',
        'Data S5': 'microbiome_filtered',
        'Data_S5': 'microbiome_filtered',
    }
    
    for sheet_name, sheet_info in report['sheets'].items():
        candidates = sheet_info['candidate_columns']
        orientation = sheet_info['orientation_guess']
        
        # Determine omic type
        omic_type = None
        for key, value in sheet_to_omic.items():
            if key.lower() in sheet_name.lower():
                omic_type = value
                break
        
        # If not found by key, try to infer from column names
        if omic_type is None:
            all_cols_lower = ' '.join(str(c).lower() for c in sheet_info['columns'])
            if 'protein' in all_cols_lower or 'peptide' in all_cols_lower:
                omic_type = 'proteomics'
            elif 'metabolite' in all_cols_lower or 'compound' in all_cols_lower:
                omic_type = 'metabolomics'
            elif 'gene' in all_cols_lower or 'transcript' in all_cols_lower:
                omic_type = 'transcriptomics'
            elif 'asv' in all_cols_lower or 'otu' in all_cols_lower or '16s' in all_cols_lower:
                omic_type = 'microbiome'
            else:
                omic_type = 'unknown'
        
        # Select best candidate columns
        sample_id_col = candidates['sample_id'][0] if candidates['sample_id'] else None
        feature_id_col = candidates['feature_id'][0] if candidates['feature_id'] else None
        
        # If no explicit sample ID column, use first column or index
        if sample_id_col is None:
            sample_id_col = str(sheet_info['columns'][0])
        
        mapping['sheets'][sheet_name] = {
            'omic_type': omic_type,
            'sample_id_column': sample_id_col,
            'feature_id_column': feature_id_col,
            'metadata_columns': candidates['metadata'],
            'value_columns': candidates['values'][:10] if len(candidates['values']) > 10 else candidates['values'],
            'value_columns_pattern': 'numeric_*' if len(candidates['values']) > 10 else None,
            'orientation': 'rows_are_samples' if orientation['rows_are_samples'] else 'rows_are_features',
            'orientation_confidence': orientation['confidence'],
            'needs_transpose': not orientation['rows_are_samples'],
        }
    
    return mapping


def extract_matrices(
    excel_path: Path,
    mapping: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Extract canonical TSV matrices from Excel workbook.
    
    Args:
        excel_path: Path to Excel file
        mapping: Column mapping configuration
        output_dir: Output directory for TSV files
        
    Returns:
        Dictionary mapping omic type to output file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    
    # Read with header=1 to skip description row in this dataset
    # The Williams et al. Excel has a description in row 0 and headers in row 1
    sheets = pd.read_excel(excel_path, sheet_name=None, header=1)
    
    for sheet_name, sheet_config in mapping['sheets'].items():
        if sheet_name not in sheets:
            logger.warning(f"Sheet '{sheet_name}' not found in workbook, skipping")
            continue
        
        df = sheets[sheet_name]
        omic_type = sheet_config['omic_type']
        
        if omic_type == 'unknown':
            logger.warning(f"Skipping sheet '{sheet_name}' with unknown omic type")
            continue
        
        logger.info(f"Extracting {omic_type} from sheet '{sheet_name}'")
        
        # For Williams et al. dataset, identify sample columns by MC-* pattern
        # Sample columns contain sample IDs like MC-289_T1-Amb_1603
        sample_pattern = r'^MC-\d+_'
        sample_cols = [c for c in df.columns if isinstance(c, str) and re.match(sample_pattern, c)]
        
        if sample_cols:
            logger.info(f"    Found {len(sample_cols)} sample columns matching MC-* pattern")
            
            # This sheet has rows=features, columns=samples
            # Need to transpose so rows=samples, columns=features
            
            # Determine the feature ID column - prefer 'Accession' for proteomics data
            # This contains the protein identifiers (e.g., Montipora_capitata_PredGene_g29386)
            if 'Accession' in df.columns:
                feature_id_col = 'Accession'
            elif sheet_config.get('feature_id_column') and sheet_config['feature_id_column'] in df.columns:
                feature_id_col = sheet_config['feature_id_column']
            else:
                # Fallback: use first column that looks like a feature identifier
                feature_id_col = df.columns[0]
            
            logger.info(f"    Using '{feature_id_col}' as feature identifier column")
            
            # Extract just sample columns and the feature ID
            if feature_id_col in df.columns:
                df_subset = df[[feature_id_col] + sample_cols].copy()
                df_subset = df_subset.set_index(feature_id_col)
            else:
                df_subset = df[sample_cols].copy()
            
            # Transpose: rows become samples, columns become features
            df_values = df_subset.T
            df_values.index.name = 'sample_id'
            
        else:
            # Standard numeric extraction
            # Handle orientation
            if sheet_config.get('needs_transpose', False):
                df = df.T
                df.columns = df.iloc[0]
                df = df.iloc[1:]
            
            # Set index to sample ID column if specified
            sample_id_col = sheet_config.get('sample_id_column')
            if sample_id_col and sample_id_col in df.columns:
                df = df.set_index(sample_id_col)
            
            # Select only numeric columns (feature values)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_values = df[numeric_cols]
        
        # Save to TSV
        output_file = output_dir / f"{omic_type}.tsv"
        df_values.to_csv(output_file, sep='\t')
        output_files[omic_type] = output_file
        
        logger.info(f"  Saved {output_file} ({df_values.shape[0]} samples x {df_values.shape[1]} features)")
        
        # Save metadata separately
        metadata_cols = sheet_config.get('metadata_columns', [])
        if metadata_cols:
            meta_df = df[[c for c in metadata_cols if c in df.columns]]
            if not meta_df.empty:
                meta_file = output_dir / f"{omic_type}_metadata.tsv"
                meta_df.to_csv(meta_file, sep='\t')
                logger.info(f"  Saved metadata to {meta_file}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract and inspect coral multi-omics Excel workbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Inspect headers and create schema report
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --report
    
    # Auto-extract using suggested mappings
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --auto --out data/processed/
    
    # Extract with confirmed mapping file
    python scripts/extract_excel.py --input data/Supplemental_Data.xlsx --mapping data/column_map_confirmed.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/Supplemental_Data.xlsx'),
        help='Path to Excel workbook',
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate schema report without extracting',
    )
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Auto-extract using suggested mappings',
    )
    parser.add_argument(
        '--mapping', '-m',
        type=Path,
        help='Path to confirmed column mapping YAML file',
    )
    parser.add_argument(
        '--out', '-o',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for extracted TSV files',
    )
    parser.add_argument(
        '--schema-out',
        type=Path,
        default=Path('data/schema_report.json'),
        help='Output path for schema report',
    )
    parser.add_argument(
        '--mapping-out',
        type=Path,
        default=Path('data/column_map_suggested.yaml'),
        help='Output path for suggested column mapping',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # Check input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("Please ensure the Excel workbook is at data/Supplemental_Data.xlsx")
        logger.info("You can download it from: DOI 10.7717/peerj.12335")
        sys.exit(1)
    
    # Inspect workbook
    logger.info("=" * 60)
    logger.info("Coral Multi-Omics Excel Extraction")
    logger.info("=" * 60)
    
    report = inspect_workbook(args.input)
    
    # Save schema report
    args.schema_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.schema_out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Schema report saved to: {args.schema_out}")
    
    # Create suggested mapping
    mapping = create_column_mapping(report)
    
    with open(args.mapping_out, 'w') as f:
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Suggested mapping saved to: {args.mapping_out}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SHEET SUMMARY")
    print("=" * 60)
    for sheet_name, sheet_info in report['sheets'].items():
        omic_type = mapping['sheets'].get(sheet_name, {}).get('omic_type', 'unknown')
        orient = mapping['sheets'].get(sheet_name, {}).get('orientation', 'unknown')
        print(f"\n{sheet_name}:")
        print(f"  Shape: {sheet_info['shape'][0]} x {sheet_info['shape'][1]}")
        print(f"  Detected omic type: {omic_type}")
        print(f"  Orientation: {orient}")
        print(f"  Sample ID column: {mapping['sheets'].get(sheet_name, {}).get('sample_id_column', 'N/A')}")
    
    # Report only mode
    if args.report and not args.auto:
        print("\n" + "=" * 60)
        print("REPORT MODE - No extraction performed")
        print("=" * 60)
        print(f"\nReview the suggested mapping at: {args.mapping_out}")
        print("To extract with auto mapping, run with --auto flag")
        print("To extract with custom mapping, run with --mapping <file.yaml>")
        return
    
    # Determine which mapping to use
    if args.mapping and args.mapping.exists():
        logger.info(f"Using provided mapping file: {args.mapping}")
        with open(args.mapping) as f:
            mapping = yaml.safe_load(f)
    elif args.auto:
        logger.info("Using auto-generated mapping")
        # Save auto-confirmed mapping
        auto_confirmed_path = Path('data/column_map_auto_confirmed.yaml')
        with open(auto_confirmed_path, 'w') as f:
            yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Auto-confirmed mapping saved to: {auto_confirmed_path}")
    else:
        logger.warning("No mapping provided and --auto not set")
        logger.info("Please review suggested mapping and re-run with --auto or --mapping")
        sys.exit(0)
    
    # Extract matrices
    print("\n" + "=" * 60)
    print("EXTRACTING MATRICES")
    print("=" * 60)
    
    output_files = extract_matrices(args.input, mapping, args.out)
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nExtracted files:")
    for omic_type, file_path in output_files.items():
        print(f"  {omic_type}: {file_path}")
    
    print(f"\nOutput directory: {args.out}")


if __name__ == '__main__':
    main()
