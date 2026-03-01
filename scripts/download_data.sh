#!/usr/bin/env bash
# =============================================================================
# Coral Holobiont Multi-Omics Data Download Script
# =============================================================================
# This script downloads real coral holobiont multi-omic data for the 
# VFL research project.
#
# Primary Dataset: Montipora capitata thermal stress multi-omics
# Source: Williams et al., PeerJ (2021) & iScience (2023)
#
# Data Sources:
#   - Zenodo: DOI 10.5281/zenodo.6861688 (processed matrices)
#   - NCBI BioProject PRJNA694677 (RNA-seq raw reads)
#   - NCBI BioProject PRJNA783340 (16S microbiome)
#   - MassIVE MSV000088443 (proteomics raw data)
#
# Usage:
#   bash scripts/download_data.sh [--auto|--manual|--minimal]
#
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_DIR="${PROJECT_ROOT}/data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
METADATA_DIR="${DATA_DIR}/metadata"

# Zenodo record for processed data
ZENODO_RECORD_ID="6861688"
ZENODO_API_URL="https://zenodo.org/api/records/${ZENODO_RECORD_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
print_header() {
    echo ""
    echo "============================================================================="
    echo "  Coral Holobiont Multi-Omics Data Downloader"
    echo "  For: Vertical Federated Learning Research Project"
    echo "============================================================================="
    echo ""
    echo "This script will help you acquire real coral holobiont multi-omic data"
    echo "from the following publications:"
    echo ""
    echo "  PRIMARY DATASET:"
    echo "  Williams et al. (2021) - 'Multi-omic characterization of the thermal"
    echo "  stress phenome in the stony coral Montipora capitata'"
    echo "  DOI: 10.7717/peerj.12335"
    echo ""
    echo "  EXTENDED DATASET:"
    echo "  Williams et al. (2023) - 'Peeling back the layers of coral holobiont"
    echo "  multi-omics data'"
    echo "  DOI: 10.1016/j.isci.2023.107623"
    echo ""
    echo "  DATA REPOSITORY:"
    echo "  Zenodo DOI: 10.5281/zenodo.6861688"
    echo ""
    echo "============================================================================="
    echo ""
}

# Create directory structure
create_directories() {
    log_info "Creating data directory structure..."
    
    mkdir -p "${RAW_DIR}"
    mkdir -p "${PROCESSED_DIR}"
    mkdir -p "${METADATA_DIR}"
    mkdir -p "${RAW_DIR}/transcriptomics"
    mkdir -p "${RAW_DIR}/metabolomics"
    mkdir -p "${RAW_DIR}/proteomics"
    mkdir -p "${RAW_DIR}/microbiome"
    
    log_success "Directory structure created at ${DATA_DIR}"
}

# Check for required tools
check_dependencies() {
    log_info "Checking required tools..."
    
    local missing_tools=()
    
    for tool in curl wget python3; do
        if ! command -v "${tool}" &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install them and re-run this script."
        exit 1
    fi
    
    log_success "All required tools are available."
}

# Download from Zenodo (automatic mode)
download_zenodo_auto() {
    log_info "Fetching file list from Zenodo record ${ZENODO_RECORD_ID}..."
    
    # Get the record metadata
    local record_json
    record_json=$(curl -s "${ZENODO_API_URL}")
    
    if [ -z "${record_json}" ]; then
        log_error "Failed to fetch Zenodo record metadata."
        return 1
    fi
    
    # Parse file URLs using Python (more reliable than jq for complex JSON)
    python3 << EOF
import json
import subprocess
import sys
import os

data = '''${record_json}'''
record = json.loads(data)

raw_dir = "${RAW_DIR}"

if 'files' not in record:
    print("ERROR: No files found in Zenodo record", file=sys.stderr)
    sys.exit(1)

files = record['files']
print(f"Found {len(files)} files in Zenodo record")

for f in files:
    filename = f['key']
    download_url = f['links']['self']
    filesize = f.get('size', 0)
    filesize_mb = filesize / (1024 * 1024)
    
    print(f"Downloading: {filename} ({filesize_mb:.1f} MB)")
    
    # Determine subdirectory based on filename
    if 'transcript' in filename.lower() or 'rna' in filename.lower() or 'gene' in filename.lower():
        subdir = 'transcriptomics'
    elif 'metab' in filename.lower() or 'metabol' in filename.lower():
        subdir = 'metabolomics'
    elif 'proteo' in filename.lower() or 'protein' in filename.lower():
        subdir = 'proteomics'
    elif 'microb' in filename.lower() or '16s' in filename.lower() or 'otu' in filename.lower():
        subdir = 'microbiome'
    else:
        subdir = ''
    
    output_path = os.path.join(raw_dir, subdir, filename) if subdir else os.path.join(raw_dir, filename)
    
    # Download using curl
    result = subprocess.run(
        ['curl', '-L', '-o', output_path, '--progress-bar', download_url],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"  -> Saved to {output_path}")
    else:
        print(f"  -> FAILED to download {filename}", file=sys.stderr)

print("\\nDownload complete!")
EOF
    
    return $?
}

# Manual download instructions
show_manual_instructions() {
    echo ""
    log_info "MANUAL DOWNLOAD INSTRUCTIONS"
    echo ""
    echo "Since automatic download was not selected (or failed), please follow"
    echo "these steps to manually acquire the data:"
    echo ""
    echo "============================================================================="
    echo "STEP 1: Download Processed Multi-Omics Data from Zenodo"
    echo "============================================================================="
    echo ""
    echo "  1. Visit: https://zenodo.org/record/6861688"
    echo "  2. Download all data files (CSV/TSV matrices)"
    echo "  3. Place them in the following directories:"
    echo ""
    echo "     Transcriptomics files -> ${RAW_DIR}/transcriptomics/"
    echo "     Metabolomics files    -> ${RAW_DIR}/metabolomics/"
    echo "     Proteomics files      -> ${RAW_DIR}/proteomics/"
    echo "     Microbiome files      -> ${RAW_DIR}/microbiome/"
    echo "     Metadata/phenotypes   -> ${METADATA_DIR}/"
    echo ""
    echo "============================================================================="
    echo "STEP 2: Expected File Structure"
    echo "============================================================================="
    echo ""
    echo "After downloading and organizing, you should have:"
    echo ""
    echo "  data/"
    echo "  ├── raw/"
    echo "  │   ├── transcriptomics/"
    echo "  │   │   └── *.csv / *.tsv (gene expression matrices)"
    echo "  │   ├── metabolomics/"
    echo "  │   │   └── *.csv / *.tsv (metabolite abundance tables)"
    echo "  │   ├── proteomics/"
    echo "  │   │   └── *.csv / *.tsv (protein abundance tables)"
    echo "  │   └── microbiome/"
    echo "  │       └── *.csv / *.tsv (OTU/ASV tables)"
    echo "  └── metadata/"
    echo "      └── sample_metadata.csv (sample IDs, conditions, color scores)"
    echo ""
    echo "============================================================================="
    echo "STEP 3: Optional - Raw Sequencing Data"
    echo "============================================================================="
    echo ""
    echo "For full reproducibility from raw reads (optional, requires significant"
    echo "compute resources):"
    echo ""
    echo "  RNA-seq (NCBI BioProject PRJNA694677):"
    echo "    wget -r ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRRXXXXXX/"
    echo ""
    echo "  16S Microbiome (NCBI BioProject PRJNA783340):"
    echo "    Use SRA Toolkit: prefetch SRRXXXXXX && fasterq-dump SRRXXXXXX"
    echo ""
    echo "  Proteomics (MassIVE MSV000088443):"
    echo "    Visit: https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000088443"
    echo ""
    echo "============================================================================="
    echo ""
    log_info "After placing files, run: python scripts/preprocess_data.py"
    echo ""
}

# Download minimal test data (for quick validation)
download_minimal() {
    log_info "Creating minimal test dataset for pipeline validation..."
    
    # Create a small synthetic subset that mimics the real data structure
    # This is ONLY for testing the pipeline, not for actual experiments
    
    python3 << EOF
import numpy as np
import pandas as pd
import os

np.random.seed(42)

raw_dir = "${RAW_DIR}"
meta_dir = "${METADATA_DIR}"

# Number of samples for minimal test
n_samples = 30

# Create sample IDs
sample_ids = [f"MCAP_{i:03d}" for i in range(n_samples)]

# Create condition labels (balanced stressed vs ambient)
conditions = ['stressed'] * (n_samples // 2) + ['ambient'] * (n_samples - n_samples // 2)
np.random.shuffle(conditions)

# Create synthetic color scores (lower = more bleached/stressed)
color_scores = np.where(
    np.array(conditions) == 'stressed',
    np.random.uniform(1, 3, n_samples),
    np.random.uniform(4, 6, n_samples)
)

# Metadata
metadata = pd.DataFrame({
    'sample_id': sample_ids,
    'condition': conditions,
    'color_score': color_scores,
    'timepoint': np.random.choice(['T0', 'T1', 'T2'], n_samples),
    'tank': np.random.choice(['A', 'B', 'C'], n_samples)
})
metadata.to_csv(os.path.join(meta_dir, 'sample_metadata.csv'), index=False)
print(f"Created: {meta_dir}/sample_metadata.csv")

# Transcriptomics (simulated gene expression - 500 genes)
n_genes = 500
gene_names = [f"gene_{i:04d}" for i in range(n_genes)]
expr_data = np.random.lognormal(mean=5, sigma=2, size=(n_samples, n_genes))
# Add condition-related signal to some genes
stress_genes = np.random.choice(n_genes, 50, replace=False)
for g in stress_genes:
    expr_data[:, g] = np.where(
        np.array(conditions) == 'stressed',
        expr_data[:, g] * 2,
        expr_data[:, g]
    )
transcriptomics_df = pd.DataFrame(expr_data, index=sample_ids, columns=gene_names)
os.makedirs(os.path.join(raw_dir, 'transcriptomics'), exist_ok=True)
transcriptomics_df.to_csv(os.path.join(raw_dir, 'transcriptomics', 'gene_counts.csv'))
print(f"Created: {raw_dir}/transcriptomics/gene_counts.csv")

# Metabolomics (simulated metabolite abundances - 100 metabolites)
n_metabolites = 100
metabolite_names = [f"metabolite_{i:03d}" for i in range(n_metabolites)]
metab_data = np.random.lognormal(mean=3, sigma=1.5, size=(n_samples, n_metabolites))
# Add condition-related signal
stress_metabs = np.random.choice(n_metabolites, 20, replace=False)
for m in stress_metabs:
    metab_data[:, m] = np.where(
        np.array(conditions) == 'stressed',
        metab_data[:, m] * 1.5,
        metab_data[:, m]
    )
metabolomics_df = pd.DataFrame(metab_data, index=sample_ids, columns=metabolite_names)
os.makedirs(os.path.join(raw_dir, 'metabolomics'), exist_ok=True)
metabolomics_df.to_csv(os.path.join(raw_dir, 'metabolomics', 'metabolite_abundances.csv'))
print(f"Created: {raw_dir}/metabolomics/metabolite_abundances.csv")

# Proteomics (simulated protein abundances - 200 proteins)
n_proteins = 200
protein_names = [f"protein_{i:03d}" for i in range(n_proteins)]
prot_data = np.random.lognormal(mean=4, sigma=1.5, size=(n_samples, n_proteins))
# Add condition-related signal
stress_prots = np.random.choice(n_proteins, 30, replace=False)
for p in stress_prots:
    prot_data[:, p] = np.where(
        np.array(conditions) == 'stressed',
        prot_data[:, p] * 1.8,
        prot_data[:, p]
    )
proteomics_df = pd.DataFrame(prot_data, index=sample_ids, columns=protein_names)
os.makedirs(os.path.join(raw_dir, 'proteomics'), exist_ok=True)
proteomics_df.to_csv(os.path.join(raw_dir, 'proteomics', 'protein_abundances.csv'))
print(f"Created: {raw_dir}/proteomics/protein_abundances.csv")

# Microbiome (simulated 16S OTU abundances - 150 OTUs)
n_otus = 150
otu_names = [f"OTU_{i:03d}" for i in range(n_otus)]
otu_data = np.random.negative_binomial(n=5, p=0.5, size=(n_samples, n_otus)).astype(float)
# Add condition-related signal
stress_otus = np.random.choice(n_otus, 25, replace=False)
for o in stress_otus:
    otu_data[:, o] = np.where(
        np.array(conditions) == 'stressed',
        otu_data[:, o] * 2,
        otu_data[:, o]
    )
microbiome_df = pd.DataFrame(otu_data, index=sample_ids, columns=otu_names)
os.makedirs(os.path.join(raw_dir, 'microbiome'), exist_ok=True)
microbiome_df.to_csv(os.path.join(raw_dir, 'microbiome', 'otu_table.csv'))
print(f"Created: {raw_dir}/microbiome/otu_table.csv")

# Create a file indicating this is synthetic test data
with open(os.path.join(raw_dir, 'SYNTHETIC_TEST_DATA.txt'), 'w') as f:
    f.write("WARNING: This directory contains SYNTHETIC test data\\n")
    f.write("This data is ONLY for pipeline validation and testing.\\n")
    f.write("For actual experiments, download real data from Zenodo DOI 10.5281/zenodo.6861688\\n")

print("\\nMinimal test dataset created successfully!")
print("WARNING: This is SYNTHETIC data for testing only. Use real data for experiments.")
EOF
    
    log_warning "Synthetic test data created. This is for pipeline testing ONLY."
    log_info "For actual experiments, please download real data using --auto or --manual mode."
}

# Main download function
download_data() {
    local mode="${1:-interactive}"
    
    case "${mode}" in
        auto)
            log_info "Automatic download mode selected."
            download_zenodo_auto
            ;;
        manual)
            show_manual_instructions
            ;;
        minimal)
            download_minimal
            ;;
        interactive)
            echo ""
            echo "Please select a download mode:"
            echo ""
            echo "  1) AUTO    - Automatically download from Zenodo (recommended)"
            echo "  2) MANUAL  - Show instructions for manual download"
            echo "  3) MINIMAL - Create synthetic test data (for pipeline testing only)"
            echo "  4) EXIT    - Exit without downloading"
            echo ""
            read -p "Enter choice [1-4]: " choice
            
            case "${choice}" in
                1)
                    download_zenodo_auto
                    ;;
                2)
                    show_manual_instructions
                    ;;
                3)
                    download_minimal
                    ;;
                4)
                    log_info "Exiting without downloading."
                    exit 0
                    ;;
                *)
                    log_error "Invalid choice. Exiting."
                    exit 1
                    ;;
            esac
            ;;
        *)
            log_error "Unknown mode: ${mode}"
            echo "Usage: $0 [--auto|--manual|--minimal]"
            exit 1
            ;;
    esac
}

# Post-download verification
verify_download() {
    log_info "Verifying downloaded data..."
    
    local has_data=false
    
    for omic in transcriptomics metabolomics proteomics microbiome; do
        if ls "${RAW_DIR}/${omic}"/*.{csv,tsv,xlsx} 2>/dev/null | head -1 > /dev/null; then
            log_success "Found ${omic} data files."
            has_data=true
        else
            log_warning "No ${omic} data files found in ${RAW_DIR}/${omic}/"
        fi
    done
    
    if [ -f "${METADATA_DIR}/sample_metadata.csv" ]; then
        log_success "Found sample metadata."
    else
        log_warning "No sample_metadata.csv found in ${METADATA_DIR}/"
    fi
    
    if [ "${has_data}" = true ]; then
        log_success "Data verification complete."
        echo ""
        log_info "Next step: Run 'python scripts/preprocess_data.py' to preprocess the data."
    else
        log_warning "No data files found. Please run with --auto or --manual mode."
    fi
}

# Main entry point
main() {
    print_header
    check_dependencies
    create_directories
    
    # Parse command line arguments
    local mode="interactive"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                mode="auto"
                shift
                ;;
            --manual)
                mode="manual"
                shift
                ;;
            --minimal)
                mode="minimal"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--auto|--manual|--minimal]"
                echo ""
                echo "Options:"
                echo "  --auto     Automatically download data from Zenodo"
                echo "  --manual   Show manual download instructions"
                echo "  --minimal  Create synthetic test data for pipeline validation"
                echo "  -h|--help  Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    download_data "${mode}"
    verify_download
}

# Run main
main "$@"
