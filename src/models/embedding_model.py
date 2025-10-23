import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.data_loader import DataLoader

class EmbeddingModel:
    def __init__(self, device: str = None):        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        self.qdrant_client = None
        self.collection_name = "topic_matching"
        
        self._init_qdrant()

    def _init_qdrant(self, host = "localhost", port = 6333):
        try:
            self.qdrant_client = QdrantClient(host=host, port=port)

            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connected successfully at {host}:{port}")
            logger.info(f"Existing collections: {len(collections.collections)}")
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}")
            logger.warning("Falling back to standard similarity computation")
            self.qdrant_client = None
    
    def encode_texts(self, texts, batch_size = None):
        if batch_size is None:
            batch_size = 32
        
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        return embeddings
    
    def index_topics_to_qdrant(self, topic_texts, topic_ids, metadata = None, recreate = True):  
        logger.info(f"Indexing {len(topic_texts)} topics to Qdrant...")
        print(topic_ids)
        
        topic_embeddings = self.encode_texts(topic_texts)
        topic_embeddings = topic_embeddings / np.linalg.norm(topic_embeddings, axis=1, keepdims=True)

        
        if recreate:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except:
                pass
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=topic_embeddings.shape[1],
                distance=Distance.COSINE
            )
        )
        logger.info(f"Collection '{self.collection_name}' created")
        
        points = []
        for i in range(len(topic_texts)):
            payload = {
                "topic_id": topic_ids[i],
                "topic_text": topic_texts[i]
            }
            
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            points.append(
                PointStruct(
                    id=i,
                    vector=topic_embeddings[i].tolist(),
                    payload=payload
                )
            )
        
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Successfully indexed {len(points)} topics to Qdrant")
    
    def match_opinions_to_topics(self, topic_texts, opinion_texts, opinion_ids = None, return_scores = True):
        logger.info(f"Matching {len(opinion_texts)} opinions to {len(topic_texts)} topics")
        return self._match_with_qdrant(opinion_texts, opinion_ids)
    
    def _match_with_qdrant(self, opinion_texts, opinion_ids = None):

        logger.info("Using Qdrant for fast matching...")
        
        logger.info("Encoding opinions...")
        opinion_embeddings = self.encode_texts(opinion_texts)

        opinion_embeddings = opinion_embeddings / np.linalg.norm(opinion_embeddings, axis=1, keepdims=True)

        threshold = 0.5
        
        results = {
            'matched_topic_ids': [],
            'similarity_scores': [],
            'top_k_matches': [],
            'confidence': [],
            'below_threshold': []
        }
        
        if opinion_ids is None:
            opinion_ids = list(range(len(opinion_texts)))
                
        for idx, opinion_embedding in enumerate(tqdm(opinion_embeddings, desc="Matching")):
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=opinion_embedding.tolist(),
                limit=5
            )
            
            top_topics = []
            for hit in search_result:
                topic_id = hit.payload['topic_id']
                score = hit.score
                top_topics.append((topic_id, score))
            
            best_topic_id = top_topics[0][0]
            best_score = top_topics[0][1]
            
            results['matched_topic_ids'].append(best_topic_id)
            results['similarity_scores'].append(float(best_score))
            results['top_k_matches'].append(top_topics)
            results['confidence'].append(float(best_score) >= threshold)
            
            if best_score < threshold:
                results['below_threshold'].append({
                    'opinion_id': opinion_ids[idx],
                    'matched_topic': best_topic_id,
                    'score': float(best_score)
                })
        
        logger.info(f"Matching completed with Qdrant")
        logger.info(f"Confident matches: {sum(results['confidence'])}/{len(opinion_texts)}")
        logger.info(f"Below threshold: {len(results['below_threshold'])}")
        
        return results
    
   
    def get_similarity_score(self, text1, text2):
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)
        
        score = cosine_similarity([emb1], [emb2])[0][0]
        return float(score)
    
    def query_similar_topics(self, query_text, top_k = 5):   
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        results = []
        for hit in search_result:
            topic_id = hit.payload['topic_id']
            topic_text = hit.payload['topic_text']
            score = hit.score
            results.append((topic_id, topic_text, score))
        
        return results


class TopicMatchingEvaluator:
    @staticmethod
    def plot_similarity_distribution(similarity_scores, output_path=None):
        
        _, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(similarity_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(similarity_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(similarity_scores):.3f}')
        ax.axvline(0.7, color='green', linestyle='--', 
                   linewidth=2, label='Threshold: 0.70')
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Topic-Opinion Similarity Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # setup_logging()
    loader = DataLoader()
    topics_df, opinions_df, _ = loader.load_all_data(clean=True)
    logger.info("Initializing embedding model with Qdrant...")
    embedding_model = EmbeddingModel()
    logger.info("Indexing topics to Qdrant...")
    embedding_model.index_topics_to_qdrant( topic_texts=topics_df['text'].tolist(), topic_ids=topics_df['topic_id'].tolist(), recreate=True )
