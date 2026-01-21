"""
AgentsCourt Similar Case Retrieval and Reranking Module
Perform semantic similarity calculation and reranking on candidate cases based on BGE semantic embedding model

Main Functions:
1. Use BGE-large-zh-v1.5 model for semantic embedding
2. Perform chunking processing on long documents to avoid length limits
3. Rerank candidate cases based on semantic similarity
4. Deduplicate and maintain priority of most relevant cases

Technical Features:
- Semantic retrieval: Beyond keyword matching, based on semantic understanding
- Document chunking: Handle ultra-long documents, improve retrieval accuracy
- Similarity calculation: Use cosine similarity for sorting
- Result optimization: Deduplicate and merge, ensure result quality

"""

from FlagEmbedding import FlagModel
import os
import json
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set GPU device, specify using GPU 0
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def document_split(cadidate_cases):
    """
    Document chunking processing function
    
    Split long documents into small chunks suitable for embedding model processing, avoid exceeding model input length limit.
    For documents exceeding 500 characters, split by 500 characters.
    
    Args:
        cadidate_cases (list): Candidate case list, each case contains unique_id and full text
        
    Returns:
        list: Chunked document list, format [{"id": unique_id, "contents": text_chunk}, ...]
        
    Processing logic:
    1. Document length <= 500 characters: Use directly
    2. Document length > 500 characters: Split by 500 characters
    3. Filter out fragments < 100 characters
    4. Keep original document ID for subsequent reorganization
    """
    id_document_list = []
    
    for entry in cadidate_cases:
        # Extract basic case information
        processed_entry = {
            "id": entry["CaseId"],  # Case unique identifier
            "contents": entry["Full Document"]   # Case full text content
        }

        # Determine if document length needs chunking
        if len(processed_entry["contents"]) > 500:
            # Chunk long documents: 500 characters per chunk
            full_texts = [processed_entry["contents"][i:i+500] 
                         for i in range(0, len(processed_entry["contents"]), 500)]
            
            # Process each chunk
            for part in full_texts:
                # Filter out too short fragments (< 100 characters) to avoid noise
                if len(part) < 100:
                    continue
                else:
                    new_entry = {"id": processed_entry["id"], "contents": part}
                    id_document_list.append(new_entry)
        else:
            # Short documents add directly, no chunking needed
            id_document_list.append(processed_entry)
            
    return id_document_list

def rerank_by_bge(query, id_document_list):
    """
    Document reranking function based on BGE semantic embedding model
    
    Use BGE model to calculate semantic similarity between query text and candidate documents,
    and sort by similarity from high to low.
    
    Args:
        query (str): Query text (detailed description of current case)
        id_document_list (list): Chunked document list, format [{"id": id, "contents": text}, ...]
        
    Returns:
        list: Unique document ID list sorted by similarity, highest similarity first
        
    Processing flow:
    1. Encode query text and all documents as vectors respectively
    2. Calculate cosine similarity between query vector and document vectors
    3. Sort by similarity descending
    4. Deduplicate and keep the highest similarity version for each document ID
    """
    # Query text list (BGE model requires input as list format)
    queries = [query]
    
    # Extract all document contents and corresponding IDs
    document_list = [li["contents"] for li in id_document_list]
    id_list = [li["id"] for li in id_document_list]
    
    # ===== Semantic Embedding Encoding =====
    # Encode query text (using dedicated query encoder)
    q_embeddings = model.encode_queries(queries)
    # Encode all candidate documents
    p_embeddings = model.encode(document_list)
    
    # ===== Similarity Calculation =====
    # Calculate cosine similarity between query vector and all document vectors (matrix multiplication)
    scores = q_embeddings @ p_embeddings.T

    # ===== Sorting and Deduplication =====
    # Pair document ID with corresponding similarity score
    paired_list = list(zip(id_list, scores[0]))
    # Sort by similarity score descending (highest similarity first)
    sorted_paired_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
    sorted_id_list = [id for id, score in sorted_paired_list]
    
    # Deduplicate: keep the highest similarity version for each ID
    # (Because the same document may be split into multiple chunks, take the most similar one)
    unique_ids = []
    for id in sorted_id_list:
        if id not in unique_ids:
            unique_ids.append(id)

    return unique_ids

def rerank_candidate(unique_ids, candidate_cases):
    """
    Reorganize candidate cases according to reranked ID list
    
    Convert the document ID list sorted by semantic similarity back to complete case object list,
    keep sorting order unchanged.
    
    Args:
        unique_ids (list): Unique document ID list sorted by similarity
        candidate_cases (list): Original candidate case list, containing complete case information
        
    Returns:
        list: Complete case list reranked by similarity
        
    Processing logic:
    1. Traverse sorted ID list
    2. Find corresponding complete case in original case list according to ID
    3. Build reranked case list in order
    4. Maintain similarity sorting order
    """
    reranked_candidates = []
    
    # Reorganize cases in similarity sorted order
    for id in unique_ids:
        # Find matching case in original candidate cases
        for case in candidate_cases:
            if case["CaseId"] == id:
                reranked_candidates.append(case)
                break  # Break out of inner loop after finding matching case
                
    return reranked_candidates
    
# ===== BGE Semantic Embedding Model Initialization =====
# Load BGE-large-zh-v1.5 Chinese semantic embedding model
model = FlagModel('model/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)  # Enable half-precision floating point to save VRAM

# ===== Data Loading Phase =====
# 1. Load candidate case data (preprocessed case database)
candidate_file_path = 'data/all.json'
with open(candidate_file_path, 'r', encoding='utf-8') as candidate_file:
    cadidate_data = json.load(candidate_file)

# 2. Load current case data to be processed
case_file_path = 'data/criminal_cases.json'
with open(case_file_path, 'r', encoding='utf-8') as case_file:
    case_data = json.load(case_file)

# ===== Main Processing Loop: Batch Reranking =====
reranked_candidates = {}  # Store reranking results for all cases

for case in tqdm(case_data, desc="案例重排序进度"):
    # Extract case ID as unique identifier
    case_id = case["案号"]

    # ===== Build Query Text: Merge Multiple Case Description Fields =====
    case_details1 = case.get("基本案情", "")    # Adjudicated facts
    case_details2 = case.get("法院意见", "")       # Judge opinion
    
    # Merge all case descriptions into complete query text
    case_details_all = case_details1 + case_details2
    # Limit query text length to avoid affecting retrieval effectiveness
    if len(case_details_all) > 2100:
        case_details_all = case_details_all[:2100]

    # ===== Retrieval and Reranking Process =====
    # 1. Get candidate case list corresponding to current case
    candidate_cases = cadidate_data

    # 2. Perform document chunking on candidate cases
    id_document_list = document_split(candidate_cases)
    
    # 3. Use BGE model to calculate semantic similarity and rerank
    unique_ids = rerank_by_bge(case_details_all, id_document_list)
    
    # 4. Reorganize candidate case list according to reranking results
    reranked_list = rerank_candidate(unique_ids, candidate_cases)
    
    # 5. Save reranking results
    reranked_candidates[case_id] = reranked_list

# ===== Save Final Results =====
# Save reranking results of all cases to JSON file
with open("reranked_candidates.json", "w", encoding='utf-8') as file:
    json.dump(reranked_candidates, file, ensure_ascii=False, indent=4)

print(f"重排序完成！处理了 {len(reranked_candidates)} 个案例的候选案例重排序。")

