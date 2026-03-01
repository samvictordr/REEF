#!/usr/bin/env python3
"""
Biological Validation for Domain Feature Rankings.

This module provides tools to validate that Domain-identified features have
biological meaning. Key validation approaches:

1. Feature Mapping: Map internal feature indices to biological identifiers
   - Protein accessions -> UniProt IDs
   - Gene names -> Gene symbols
   - Metabolite peaks -> Known metabolites

2. Pathway Enrichment: Test if top domain features are enriched in stress-related pathways
   - Heat shock response
   - Oxidative stress
   - Protein degradation
   - Coral-specific pathways

3. Literature Validation: Cross-reference with known coral stress biomarkers
   - Heat shock proteins (HSP70, HSP90)
   - Antioxidant enzymes (SOD, catalase)
   - Apoptosis markers

4. Functional Annotation: Annotate top features with GO terms, KEGG pathways

Reference:
Williams et al. coral heat stress dataset - we expect top features to relate
to heat shock response, oxidative stress, and immune/apoptosis pathways.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Known Coral Stress Biomarkers (Literature Curated)
# =============================================================================

# Key coral stress-related proteins and genes from literature
KNOWN_STRESS_MARKERS = {
    # Heat Shock Proteins
    "HSP70": ["hsp70", "heat shock protein 70", "hspa", "dnak"],
    "HSP90": ["hsp90", "heat shock protein 90", "hsp83", "htpg"],
    "HSP60": ["hsp60", "chaperonin", "groEL"],
    "sHSP": ["small heat shock", "hsp20", "hsp22", "alpha crystallin"],
    
    # Oxidative Stress Response
    "SOD": ["superoxide dismutase", "sod1", "sod2", "mnsod", "cu/zn sod"],
    "Catalase": ["catalase", "cat"],
    "GPx": ["glutathione peroxidase", "gpx"],
    "Thioredoxin": ["thioredoxin", "trx", "peroxiredoxin", "prdx"],
    
    # Protein Degradation / Ubiquitin
    "Ubiquitin": ["ubiquitin", "polyubiquitin", "ub"],
    "Proteasome": ["proteasome", "psm", "psmb", "psmc"],
    
    # Apoptosis / Cell Death
    "Bcl2": ["bcl-2", "bcl2", "bax", "apoptosis regulator"],
    "Caspase": ["caspase", "casp"],
    
    # Immune Response
    "Lectin": ["lectin", "c-type lectin", "mannose binding"],
    "Complement": ["complement", "c3", "factor b"],
    
    # Symbiodinium-related
    "Photosynthesis": ["photosystem", "rubisco", "light harvesting", "chlorophyll"],
    "Carbonic_anhydrase": ["carbonic anhydrase"],
}

# GO Terms associated with stress response
STRESS_GO_TERMS = {
    "GO:0006950": "response to stress",
    "GO:0009408": "response to heat",
    "GO:0006457": "protein folding",
    "GO:0006979": "response to oxidative stress",
    "GO:0006915": "apoptotic process",
    "GO:0006955": "immune response",
    "GO:0055114": "oxidation-reduction process",
    "GO:0051082": "unfolded protein binding",
}

# KEGG Pathways for stress
STRESS_KEGG_PATHWAYS = {
    "ko04141": "Protein processing in endoplasmic reticulum",
    "ko04210": "Apoptosis",
    "ko04610": "Complement and coagulation cascades",
    "ko00190": "Oxidative phosphorylation",
    "ko04612": "Antigen processing and presentation",
}


# =============================================================================
# Williams et al. (2021) Validated Stress Response Proteins
# DOI: 10.7717/peerj.12335
# =============================================================================

# These are proteins specifically identified by Williams et al. (2021) as
# differentially expressed under thermal stress in Montipora capitata.
# They serve as ground-truth validation for our domain feature rankings.

WILLIAMS_STRESS_PROTEINS = {
    # Heat Shock Response (Table S3, S4 in Williams et al.)
    "HSP70_family": [
        "heat shock cognate 71", "heat shock 70 kDa", "hsp70", 
        "DnaK", "heat shock cognate protein", "molecular chaperone HSP70",
    ],
    "HSP90_family": [
        "hsp90", "heat shock protein 90", "hsp83", "endoplasmin",
        "90 kDa heat shock", "hsp90-alpha", "hsp90-beta",
    ],
    "Small_HSPs": [
        "small heat shock protein", "hsp20", "crystallin", "hsp27",
        "alpha-crystallin", "heat shock protein beta",
    ],
    "Chaperonins": [
        "chaperonin", "t-complex protein", "tcp1", "cct", 
        "60 kDa heat shock", "hsp60", "groEL",
    ],
    
    # Oxidative Stress Response (Williams et al. Table S5)
    "Antioxidant_enzymes": [
        "superoxide dismutase", "catalase", "glutathione peroxidase",
        "peroxiredoxin", "thioredoxin", "glutathione s-transferase",
        "ferritin", "manganese superoxide", "copper/zinc superoxide",
    ],
    "Redox_signaling": [
        "thioredoxin-dependent", "glutaredoxin", "sulfiredoxin",
        "methionine sulfoxide reductase", "protein disulfide-isomerase",
    ],
    
    # Protein Homeostasis (Williams et al. Table S6)
    "Ubiquitin_proteasome": [
        "ubiquitin", "proteasome", "ubiquitin-conjugating",
        "26S proteasome", "ubiquitin-protein ligase", "e3 ubiquitin",
    ],
    "ER_stress": [
        "endoplasmic reticulum", "calreticulin", "calnexin",
        "protein disulfide", "unfolded protein", "ER stress",
    ],
    
    # Apoptosis (Williams et al. identified apoptosis markers)
    "Apoptosis_regulators": [
        "caspase", "bcl-2", "bax", "apoptosis-inducing",
        "programmed cell death", "death-associated", "apaf",
    ],
    "DNA_damage_response": [
        "p53", "dna repair", "rad51", "xrcc", "damage-inducible",
    ],
    
    # Immune/Innate defense (Williams et al. Table S7)
    "Innate_immunity": [
        "lectin", "complement c3", "toll-like", "nf-kappa",
        "antimicrobial", "lysozyme", "defensin",
    ],
    "Pattern_recognition": [
        "scavenger receptor", "mannose-binding", "fibrinogen-related",
        "peptidoglycan recognition", "galectin",
    ],
    
    # Cytoskeleton/Cell structure (stress remodeling)
    "Cytoskeleton": [
        "actin", "tubulin", "myosin", "spectrin", "filamin",
        "tropomyosin", "cofilin",
    ],
    
    # Energy metabolism (stress energetics)
    "Energy_metabolism": [
        "atp synthase", "cytochrome c oxidase", "nadh dehydrogenase",
        "electron transport", "mitochondrial", "oxidative phosphorylation",
    ],
    
    # Symbiodinium-related (holobiont context)
    "Symbiosis": [
        "photosystem", "chlorophyll", "rubisco", "light-harvesting",
        "carbonic anhydrase", "coral symbiont", "zooxanthellae",
    ],
}

# Flatten for quick lookup
WILLIAMS_FLAT_PATTERNS = []
for category, patterns in WILLIAMS_STRESS_PROTEINS.items():
    for p in patterns:
        WILLIAMS_FLAT_PATTERNS.append((p.lower(), category))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureAnnotation:
    """Biological annotation for a feature."""
    feature_index: int
    omic_type: str
    raw_id: str
    gene_name: Optional[str] = None
    uniprot_id: Optional[str] = None
    description: Optional[str] = None
    go_terms: List[str] = field(default_factory=list)
    kegg_pathways: List[str] = field(default_factory=list)
    stress_category: Optional[str] = None
    is_known_stress_marker: bool = False


@dataclass
class EnrichmentResult:
    """Result of pathway/term enrichment analysis."""
    category: str
    term_id: str
    term_name: str
    genes_in_term: int
    genes_in_query: int
    overlap: int
    fold_enrichment: float
    p_value: float
    genes: List[str]


@dataclass
class ValidationReport:
    """Complete biological validation report."""
    n_features_analyzed: int
    n_mapped_features: int
    n_stress_markers_found: int
    stress_marker_hits: Dict[str, List[str]]
    enrichment_results: List[EnrichmentResult]
    top_annotated_features: List[FeatureAnnotation]
    validation_score: float  # 0-1 score for biological validity
    summary: str


# =============================================================================
# Feature Mapping
# =============================================================================

class FeatureMapper:
    """
    Map internal feature indices to biological identifiers.
    
    For each omic type:
    - Proteomics: Feature names are protein accessions (e.g., Montipora_capitata_PredGene_g29386)
    - Transcriptomics: Gene IDs or transcript IDs
    - Metabolomics: m/z values or metabolite names
    - Microbiome: OTU/ASV IDs or taxa names
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.feature_maps: Dict[str, pd.Index] = {}
        self._load_feature_names()
    
    def _load_feature_names(self):
        """Load feature names from processed data files."""
        processed_dir = self.data_dir / "processed"
        
        omic_files = {
            "proteomics": "proteomics.tsv",
            "transcriptomics": "transcriptomics.tsv",
            "metabolomics": "metabolomics.tsv",
            "microbiome": "microbiome.tsv",
        }
        
        for omic_name, filename in omic_files.items():
            filepath = processed_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, sep="\t", index_col=0, nrows=0)
                    self.feature_maps[omic_name] = df.columns
                    logger.info(f"Loaded {len(df.columns)} {omic_name} feature names")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
    
    def get_feature_name(self, omic_type: str, feature_index: int) -> Optional[str]:
        """Get the biological name for a feature index."""
        if omic_type not in self.feature_maps:
            return None
        
        features = self.feature_maps[omic_type]
        if feature_index < 0 or feature_index >= len(features):
            return None
        
        return features[feature_index]
    
    def get_all_feature_names(self, omic_type: str) -> Optional[pd.Index]:
        """Get all feature names for an omic type."""
        return self.feature_maps.get(omic_type)
    
    def map_domain_features(
        self, 
        domain_values: Dict[str, np.ndarray],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[int, str, float]]]:
        """
        Map top domain features to biological names.
        
        Args:
            domain_values: Dict of omic_name -> domain importance array
            top_k: Number of top features per omic to map
        
        Returns:
            Dict of omic_name -> List of (index, name, importance)
        """
        mapped_features = {}
        
        for omic_name, values in domain_values.items():
            # Get top-k indices by absolute domain value
            if isinstance(values, np.ndarray) and len(values) > 0:
                # Handle both 1D and 2D arrays
                if values.ndim > 1:
                    values = np.abs(values).mean(axis=0)
                
                top_indices = np.argsort(np.abs(values))[::-1][:top_k]
                
                mapped = []
                for idx in top_indices:
                    name = self.get_feature_name(omic_name, idx)
                    importance = float(values[idx])
                    mapped.append((int(idx), name or f"feature_{idx}", importance))
                
                mapped_features[omic_name] = mapped
        
        return mapped_features


# =============================================================================
# Stress Marker Detection
# =============================================================================

class StressMarkerDetector:
    """Detect known stress markers in feature names."""
    
    def __init__(self, custom_markers: Optional[Dict[str, List[str]]] = None):
        self.markers = KNOWN_STRESS_MARKERS.copy()
        if custom_markers:
            self.markers.update(custom_markers)
    
    def detect_marker(self, feature_name: str) -> Optional[str]:
        """
        Check if a feature name matches known stress markers.
        
        Returns marker category name if matched, else None.
        """
        if not feature_name:
            return None
        
        name_lower = feature_name.lower()
        
        for category, patterns in self.markers.items():
            for pattern in patterns:
                if pattern.lower() in name_lower:
                    return category
        
        return None
    
    def scan_features(
        self, 
        features: List[Tuple[int, str, float]],
    ) -> Dict[str, List[Tuple[int, str, float]]]:
        """
        Scan a list of features for stress markers.
        
        Args:
            features: List of (index, name, importance) tuples
        
        Returns:
            Dict of marker_category -> List of matching features
        """
        hits = defaultdict(list)
        
        for idx, name, importance in features:
            category = self.detect_marker(name)
            if category:
                hits[category].append((idx, name, importance))
        
        return dict(hits)
    
    def count_stress_markers(
        self, 
        feature_names: List[str],
    ) -> Tuple[int, Dict[str, int]]:
        """
        Count how many features match stress markers.
        
        Returns:
            (total_count, per_category_counts)
        """
        category_counts = defaultdict(int)
        
        for name in feature_names:
            category = self.detect_marker(name)
            if category:
                category_counts[category] += 1
        
        return sum(category_counts.values()), dict(category_counts)


# =============================================================================
# Williams et al. (2021) Cross-Validation
# =============================================================================

class WilliamsCrossValidator:
    """
    Cross-validate domain features against Williams et al. (2021) findings.
    
    Williams et al. performed thermal stress experiments on Montipora capitata
    and identified differentially expressed proteins, genes, and metabolites.
    This class checks if our Domain-identified features overlap with their
    validated stress response markers.
    
    Reference: Williams et al. (2021) PeerJ, DOI: 10.7717/peerj.12335
    """
    
    def __init__(
        self,
        supplemental_data_path: Optional[Path] = None,
        use_curated_list: bool = True,
    ):
        """
        Initialize cross-validator.
        
        Args:
            supplemental_data_path: Path to Williams et al. Supplemental_Data.xlsx
            use_curated_list: If True, use the curated WILLIAMS_STRESS_PROTEINS dict
        """
        self.validated_proteins = WILLIAMS_STRESS_PROTEINS.copy()
        self.supplemental_data_path = supplemental_data_path
        
        # Load additional proteins from Excel if available
        if supplemental_data_path and Path(supplemental_data_path).exists():
            self._load_supplemental_data(supplemental_data_path)
    
    def _load_supplemental_data(self, path: Path):
        """Load differentially expressed features from Williams et al. Excel file."""
        try:
            import openpyxl
            
            xl = pd.ExcelFile(path)
            
            # Expected sheets based on typical multi-omics supplemental format
            expected_sheets = {
                'proteomics': ['protein', 'prot', 'proteome'],
                'transcriptomics': ['transcript', 'gene', 'rna', 'expression'],
                'metabolomics': ['metabol', 'metab', 'compound'],
            }
            
            for omic_type, keywords in expected_sheets.items():
                for sheet_name in xl.sheet_names:
                    if any(kw in sheet_name.lower() for kw in keywords):
                        try:
                            df = pd.read_excel(xl, sheet_name=sheet_name)
                            
                            # Find ID/name column
                            id_cols = [c for c in df.columns if any(
                                kw in c.lower() for kw in ['id', 'name', 'gene', 'protein', 'accession']
                            )]
                            
                            if id_cols:
                                ids = df[id_cols[0]].dropna().astype(str).tolist()
                                key = f"Williams_{omic_type}"
                                self.validated_proteins[key] = [
                                    id_.lower() for id_ in ids[:500]  # Limit for efficiency
                                ]
                                logger.info(f"Loaded {len(ids)} IDs from {sheet_name}")
                        except Exception as e:
                            logger.warning(f"Could not parse sheet {sheet_name}: {e}")
        except ImportError:
            logger.warning("openpyxl not installed - using curated list only")
        except Exception as e:
            logger.warning(f"Could not load supplemental data: {e}")
    
    def detect_williams_marker(self, feature_name: str) -> Optional[Tuple[str, str]]:
        """
        Check if a feature matches Williams et al. validated proteins.
        
        Returns:
            Tuple of (category, matched_pattern) if found, else None
        """
        if not feature_name:
            return None
        
        name_lower = feature_name.lower()
        
        # Check against curated patterns
        for pattern, category in WILLIAMS_FLAT_PATTERNS:
            if pattern in name_lower:
                return (category, pattern)
        
        # Check against loaded supplemental data
        for category, patterns in self.validated_proteins.items():
            if category.startswith("Williams_"):
                for pattern in patterns:
                    if pattern in name_lower or name_lower in pattern:
                        return (category, pattern)
        
        return None
    
    def cross_validate_features(
        self,
        top_features: Dict[str, List[Tuple[int, str, float]]],
    ) -> Dict[str, Any]:
        """
        Cross-validate Domain-selected features with Williams et al. findings.
        
        Args:
            top_features: Dict of omic_name -> List of (index, name, importance)
        
        Returns:
            Dict with validation results including overlap counts and annotations
        """
        results = {
            "total_features": 0,
            "williams_matches": 0,
            "category_breakdown": defaultdict(list),
            "omic_breakdown": {},
            "matched_features": [],
            "unmatched_features": [],
        }
        
        for omic_name, features in top_features.items():
            omic_matches = 0
            omic_results = []
            
            for idx, name, importance in features:
                results["total_features"] += 1
                match = self.detect_williams_marker(name)
                
                if match:
                    category, pattern = match
                    results["williams_matches"] += 1
                    omic_matches += 1
                    
                    results["category_breakdown"][category].append({
                        "omic": omic_name,
                        "feature": name,
                        "importance": importance,
                        "matched_pattern": pattern,
                    })
                    
                    results["matched_features"].append({
                        "omic": omic_name,
                        "index": idx,
                        "name": name,
                        "importance": importance,
                        "williams_category": category,
                    })
                else:
                    results["unmatched_features"].append({
                        "omic": omic_name,
                        "index": idx,
                        "name": name,
                        "importance": importance,
                    })
            
            results["omic_breakdown"][omic_name] = {
                "total": len(features),
                "matches": omic_matches,
                "match_rate": omic_matches / max(len(features), 1),
            }
        
        # Convert defaultdict to regular dict
        results["category_breakdown"] = dict(results["category_breakdown"])
        
        # Compute summary statistics
        results["overall_match_rate"] = (
            results["williams_matches"] / max(results["total_features"], 1)
        )
        
        # Compute category distribution
        category_counts = {
            cat: len(features) 
            for cat, features in results["category_breakdown"].items()
        }
        results["category_counts"] = category_counts
        
        return results
    
    def generate_validation_report(
        self,
        cross_val_results: Dict[str, Any],
    ) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "WILLIAMS ET AL. (2021) CROSS-VALIDATION REPORT",
            "DOI: 10.7717/peerj.12335",
            "=" * 60,
            "",
            f"Total Domain-selected features analyzed: {cross_val_results['total_features']}",
            f"Features matching Williams et al. markers: {cross_val_results['williams_matches']}",
            f"Overall match rate: {cross_val_results['overall_match_rate']:.1%}",
            "",
            "=" * 60,
            "MATCHES BY OMIC TYPE",
            "=" * 60,
        ]
        
        for omic, stats in cross_val_results["omic_breakdown"].items():
            lines.append(f"  {omic}: {stats['matches']}/{stats['total']} ({stats['match_rate']:.1%})")
        
        lines.extend([
            "",
            "=" * 60,
            "MATCHES BY STRESS CATEGORY",
            "=" * 60,
        ])
        
        sorted_cats = sorted(
            cross_val_results["category_counts"].items(),
            key=lambda x: -x[1]
        )
        for cat, count in sorted_cats:
            lines.append(f"  {cat}: {count} features")
        
        lines.extend([
            "",
            "=" * 60,
            "TOP MATCHED FEATURES",
            "=" * 60,
        ])
        
        # Show top 20 matched features by importance
        matched = sorted(
            cross_val_results["matched_features"],
            key=lambda x: -abs(x["importance"])
        )[:20]
        
        for feat in matched:
            lines.append(
                f"  [{feat['omic']}] {feat['name'][:40]} "
                f"(importance={feat['importance']:.4f}, category={feat['williams_category']})"
            )
        
        # Interpretation
        match_rate = cross_val_results["overall_match_rate"]
        lines.extend([
            "",
            "=" * 60,
            "INTERPRETATION",
            "=" * 60,
        ])
        
        if match_rate > 0.3:
            lines.append(
                "STRONG VALIDATION: >30% of domain features match Williams et al. stress markers."
            )
            lines.append(
                "The model has successfully identified biologically relevant stress response features."
            )
        elif match_rate > 0.15:
            lines.append(
                "MODERATE VALIDATION: 15-30% of domain features match Williams et al. markers."
            )
            lines.append(
                "The model shows partial alignment with known stress biology."
            )
        else:
            lines.append(
                "WEAK VALIDATION: <15% of domain features match Williams et al. markers."
            )
            lines.append(
                "The model may be capturing novel signals or technical artifacts."
            )
        
        return "\n".join(lines)


# =============================================================================
# Simple Enrichment Analysis
# =============================================================================

def hypergeometric_test(
    overlap: int,
    query_size: int,
    term_size: int,
    background_size: int,
) -> float:
    """
    Compute hypergeometric p-value for enrichment.
    
    Tests if the overlap between query and term is greater than expected by chance.
    """
    from scipy import stats
    
    # P(X >= overlap) where X ~ Hypergeometric(N, K, n)
    # N = background_size, K = term_size, n = query_size
    p_value = stats.hypergeom.sf(
        overlap - 1,  # sf computes P(X > k), so we use k-1 for P(X >= k)
        background_size,
        term_size,
        query_size,
    )
    
    return p_value


def compute_stress_enrichment(
    top_features: List[str],
    all_features: List[str],
    detector: StressMarkerDetector,
) -> List[EnrichmentResult]:
    """
    Compute enrichment of stress markers in top features vs background.
    
    Tests if stress-related features are over-represented in the Domain-selected features.
    """
    n_top = len(top_features)
    n_all = len(all_features)
    
    if n_top == 0 or n_all == 0:
        return []
    
    results = []
    
    # Count stress markers in each set
    top_hits_count, top_per_category = detector.count_stress_markers(top_features)
    all_hits_count, all_per_category = detector.count_stress_markers(all_features)
    
    # Overall stress marker enrichment
    if all_hits_count > 0:
        expected = (all_hits_count / n_all) * n_top
        fold_enrichment = top_hits_count / expected if expected > 0 else 0
        
        p_value = hypergeometric_test(
            top_hits_count, n_top, all_hits_count, n_all
        )
        
        results.append(EnrichmentResult(
            category="Overall",
            term_id="stress_markers",
            term_name="Known Stress Markers",
            genes_in_term=all_hits_count,
            genes_in_query=n_top,
            overlap=top_hits_count,
            fold_enrichment=fold_enrichment,
            p_value=p_value,
            genes=[f for f in top_features if detector.detect_marker(f)],
        ))
    
    # Per-category enrichment
    for category in all_per_category:
        if all_per_category[category] == 0:
            continue
        
        top_count = top_per_category.get(category, 0)
        all_count = all_per_category[category]
        
        expected = (all_count / n_all) * n_top
        fold_enrichment = top_count / expected if expected > 0 else 0
        
        p_value = hypergeometric_test(
            top_count, n_top, all_count, n_all
        )
        
        results.append(EnrichmentResult(
            category="Stress Category",
            term_id=category,
            term_name=category.replace("_", " "),
            genes_in_term=all_count,
            genes_in_query=n_top,
            overlap=top_count,
            fold_enrichment=fold_enrichment,
            p_value=p_value,
            genes=[],  # Could fill in specific genes
        ))
    
    # Sort by p-value
    results.sort(key=lambda x: x.p_value)
    
    return results


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_domain_biological_relevance(
    domain_results_path: Path,
    data_dir: Path,
    top_k: int = 100,
    output_dir: Optional[Path] = None,
) -> ValidationReport:
    """
    Validate biological relevance of domain feature rankings.
    
    This is the main function to assess whether Domain-identified features
    have biological meaning in the context of coral stress response.
    
    Args:
        domain_results_path: Path to domain saliency analysis results JSON
        data_dir: Path to data directory with processed files
        top_k: Number of top features to analyze
        output_dir: Optional directory to save validation report
    
    Returns:
        ValidationReport with biological validation results
    """
    logger.info("=" * 60)
    logger.info("DOMAIN BIOLOGICAL VALIDATION")
    logger.info("=" * 60)
    
    # Load domain results
    with open(domain_results_path, "r") as f:
        domain_results = json.load(f)
    
    # Initialize tools
    mapper = FeatureMapper(data_dir)
    detector = StressMarkerDetector()
    
    # Handle different domain result formats
    mapped_features = {}
    
    # Format 1: From domain_sensitivity/summary.json - "proteomics_1969" style
    if "results_by_k" in domain_results:
        logger.info("Detected domain_sensitivity format")
        # Use the largest k value for analysis
        k_values = list(domain_results["results_by_k"].keys())
        k_to_use = max(k_values, key=int) if k_values else None
        
        if k_to_use:
            result_data = domain_results["results_by_k"][k_to_use]
            top_features = result_data.get("top_20_features", [])
            top_scores = result_data.get("top_20_scores", [])
            
            # Parse "omic_index" format
            for feat_str, score in zip(top_features, top_scores):
                parts = feat_str.rsplit("_", 1)
                if len(parts) == 2:
                    omic_name, idx_str = parts
                    try:
                        idx = int(idx_str)
                        feature_name = mapper.get_feature_name(omic_name, idx)
                        
                        if omic_name not in mapped_features:
                            mapped_features[omic_name] = []
                        mapped_features[omic_name].append(
                            (idx, feature_name or feat_str, float(score))
                        )
                    except ValueError:
                        continue
    
    # Format 2: From outputs/domain_vfl/metrics.json - importance_summary with indices
    elif "importance_summary" in domain_results:
        logger.info("Detected domain_vfl metrics format")
        for omic_name, omic_data in domain_results["importance_summary"].items():
            if "top_20_indices" in omic_data:
                indices = omic_data["top_20_indices"]
                # Flatten nested lists if needed
                if indices and isinstance(indices[0], list):
                    indices = [i[0] for i in indices if i]
                
                features = []
                for i, idx in enumerate(indices[:top_k]):
                    feature_name = mapper.get_feature_name(omic_name, idx)
                    # Use position as importance proxy
                    importance = 1.0 - (i / len(indices))
                    features.append((int(idx), feature_name or f"feature_{idx}", importance))
                
                mapped_features[omic_name] = features
    
    # Format 3: Direct feature_importance arrays
    elif any(isinstance(v, dict) and "feature_importance" in v for v in domain_results.values() if isinstance(v, dict)):
        logger.info("Detected direct feature_importance format")
        for omic_name, omic_data in domain_results.items():
            if isinstance(omic_data, dict) and "feature_importance" in omic_data:
                values = np.array(omic_data["feature_importance"])
                top_indices = np.argsort(np.abs(values))[::-1][:top_k]
                
                features = []
                for idx in top_indices:
                    feature_name = mapper.get_feature_name(omic_name, int(idx))
                    features.append((int(idx), feature_name or f"feature_{idx}", float(values[idx])))
                
                mapped_features[omic_name] = features
    
    # Format 4: Try omic_stats as fallback
    elif "omic_stats" in domain_results:
        logger.info("Using omic_stats as fallback (limited feature info)")
        for omic_name, omic_data in domain_results["omic_stats"].items():
            if isinstance(omic_data, dict):
                n_selected = omic_data.get("selected_features", 0)
                mapped_features[omic_name] = []
                # We don't have specific indices, just note the omic
    
    if not mapped_features:
        logger.warning("Could not extract feature information from domain results")
        return ValidationReport(
            n_features_analyzed=0,
            n_mapped_features=0,
            n_stress_markers_found=0,
            stress_marker_hits={},
            enrichment_results=[],
            top_annotated_features=[],
            validation_score=0.0,
            summary="Could not extract features from domain results file.",
        )
    
    # Collect all top feature names across omics
    all_top_names = []
    all_background_names = []
    
    for omic_name, features in mapped_features.items():
        for idx, name, importance in features:
            all_top_names.append(name)
        
        # Get background features
        all_names = mapper.get_all_feature_names(omic_name)
        if all_names is not None:
            all_background_names.extend(list(all_names))
    
    logger.info(f"Analyzing {len(all_top_names)} top features across {len(mapped_features)} omics")
    
    # Detect stress markers
    all_stress_hits = {}
    for omic_name, features in mapped_features.items():
        hits = detector.scan_features(features)
        for category, feature_list in hits.items():
            if category not in all_stress_hits:
                all_stress_hits[category] = []
            all_stress_hits[category].extend(
                [f"{omic_name}:{name}" for _, name, _ in feature_list]
            )
    
    n_stress_markers = sum(len(v) for v in all_stress_hits.values())
    logger.info(f"Found {n_stress_markers} known stress markers in top features")
    
    # Compute enrichment
    enrichment_results = compute_stress_enrichment(
        all_top_names,
        all_background_names,
        detector,
    )
    
    # Log significant enrichments
    for result in enrichment_results:
        if result.p_value < 0.05:
            logger.info(
                f"  {result.term_name}: fold={result.fold_enrichment:.2f}, "
                f"p={result.p_value:.4f}, overlap={result.overlap}"
            )
    
    # === Williams et al. (2021) Cross-Validation ===
    logger.info("\n" + "=" * 60)
    logger.info("WILLIAMS ET AL. (2021) CROSS-VALIDATION")
    logger.info("=" * 60)
    
    supplemental_path = data_dir / "Supplemental_Data.xlsx"
    williams_validator = WilliamsCrossValidator(
        supplemental_data_path=supplemental_path if supplemental_path.exists() else None
    )
    
    williams_results = williams_validator.cross_validate_features(mapped_features)
    williams_report_text = williams_validator.generate_validation_report(williams_results)
    logger.info("\n" + williams_report_text)
    
    # Create annotated features list
    top_annotated = []
    for omic_name, features in mapped_features.items():
        for idx, name, importance in features[:20]:  # Top 20 per omic
            stress_cat = detector.detect_marker(name)
            annotation = FeatureAnnotation(
                feature_index=idx,
                omic_type=omic_name,
                raw_id=name,
                stress_category=stress_cat,
                is_known_stress_marker=stress_cat is not None,
            )
            top_annotated.append(annotation)
    
    # Sort by whether they're stress markers (markers first)
    top_annotated.sort(key=lambda x: (not x.is_known_stress_marker, x.omic_type))
    
    # Compute validation score (0-1)
    # Higher score = more biologically relevant
    score_components = []
    
    # Component 1: Fraction of top features that are stress markers
    if all_top_names:
        stress_fraction = n_stress_markers / len(all_top_names)
        score_components.append(min(stress_fraction * 5, 1.0))  # Cap at 1.0
    
    # Component 2: Enrichment significance
    if enrichment_results:
        best_pval = min(r.p_value for r in enrichment_results)
        enrichment_score = 1 - min(best_pval, 1.0)
        score_components.append(enrichment_score)
    
    # Component 3: Diversity of stress categories
    n_categories = len(all_stress_hits)
    category_score = min(n_categories / 5, 1.0)  # Expect ~5 categories
    score_components.append(category_score)
    
    # Component 4: Williams et al. cross-validation match rate
    williams_match_score = min(williams_results["overall_match_rate"] * 3, 1.0)  # Scale: 33% = 1.0
    score_components.append(williams_match_score)
    
    validation_score = np.mean(score_components) if score_components else 0.0
    
    # Generate summary
    summary_parts = [
        f"Analyzed {len(all_top_names)} top domain features across {len(mapped_features)} omics.",
        f"Found {n_stress_markers} known stress biomarkers ({100*n_stress_markers/max(len(all_top_names),1):.1f}%).",
        f"Stress categories detected: {', '.join(all_stress_hits.keys()) if all_stress_hits else 'None'}.",
        f"Williams et al. (2021) match rate: {williams_results['overall_match_rate']:.1%} ({williams_results['williams_matches']}/{williams_results['total_features']} features).",
    ]
    
    if enrichment_results and enrichment_results[0].p_value < 0.05:
        summary_parts.append(
            f"Significant enrichment detected (p={enrichment_results[0].p_value:.4f}, "
            f"fold={enrichment_results[0].fold_enrichment:.2f})."
        )
    
    summary_parts.append(f"Biological validation score: {validation_score:.3f}")
    
    if validation_score > 0.5:
        summary_parts.append("CONCLUSION: Strong biological relevance - domain features relate to stress response.")
    elif validation_score > 0.2:
        summary_parts.append("CONCLUSION: Moderate biological relevance - some stress markers detected.")
    else:
        summary_parts.append("CONCLUSION: Limited biological relevance - few stress markers in top features.")
    
    summary = "\n".join(summary_parts)
    logger.info("\n" + summary)
    
    # Create report
    report = ValidationReport(
        n_features_analyzed=len(all_top_names),
        n_mapped_features=len([n for n in all_top_names if n and not n.startswith("feature_")]),
        n_stress_markers_found=n_stress_markers,
        stress_marker_hits={k: v for k, v in all_stress_hits.items()},
        enrichment_results=enrichment_results,
        top_annotated_features=top_annotated[:50],  # Top 50
        validation_score=validation_score,
        summary=summary,
    )
    
    # Save report if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_dict = {
            "n_features_analyzed": report.n_features_analyzed,
            "n_mapped_features": report.n_mapped_features,
            "n_stress_markers_found": report.n_stress_markers_found,
            "validation_score": report.validation_score,
            "stress_marker_hits": report.stress_marker_hits,
            "enrichment_results": [
                {
                    "term_id": r.term_id,
                    "term_name": r.term_name,
                    "fold_enrichment": r.fold_enrichment,
                    "p_value": r.p_value,
                    "overlap": r.overlap,
                }
                for r in report.enrichment_results
            ],
            "top_annotated_features": [
                {
                    "omic": a.omic_type,
                    "index": a.feature_index,
                    "name": a.raw_id,
                    "stress_category": a.stress_category,
                    "is_stress_marker": a.is_known_stress_marker,
                }
                for a in report.top_annotated_features
            ],
            # Williams et al. (2021) cross-validation results
            "williams_cross_validation": {
                "total_features": williams_results["total_features"],
                "williams_matches": williams_results["williams_matches"],
                "overall_match_rate": williams_results["overall_match_rate"],
                "omic_breakdown": williams_results["omic_breakdown"],
                "category_counts": williams_results["category_counts"],
                "top_matched_features": williams_results["matched_features"][:50],
            },
            "summary": report.summary,
        }
        
        report_path = output_dir / "biological_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Report saved to {report_path}")
        
        # Also save human-readable summary
        summary_path = output_dir / "biological_validation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DOMAIN BIOLOGICAL VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(summary + "\n\n")
            f.write("=" * 60 + "\n")
            f.write("STRESS MARKER HITS\n")
            f.write("=" * 60 + "\n")
            for category, features in report.stress_marker_hits.items():
                f.write(f"\n{category}:\n")
                for feat in features[:10]:
                    f.write(f"  - {feat}\n")
            
            # Add Williams validation report
            f.write("\n\n")
            f.write(williams_report_text)
        
        logger.info(f"Summary saved to {summary_path}")
    
    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for biological validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate biological relevance of domain feature rankings"
    )
    parser.add_argument(
        "--domain-results",
        type=str,
        required=True,
        help="Path to domain results JSON file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save validation report"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top features to analyze per omic"
    )
    
    args = parser.parse_args()
    
    report = validate_domain_biological_relevance(
        domain_results_path=Path(args.domain_results),
        data_dir=Path(args.data_dir),
        top_k=args.top_k,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    
    print(f"\nValidation Score: {report.validation_score:.3f}")


if __name__ == "__main__":
    main()
