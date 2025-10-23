import pandas as pd
from loguru import logger
import yaml
from transformers import pipeline
from src.utils import data_loader

class ConclusionGenerator:
    def __init__(self, config_path = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        logger.info(" HuggingFace Bart initialized")
    
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def analyze_opinions_by_topic(self, opinions_df):
        logger.info("Analyzing opinions by topic...")
        
        topic_analysis = {}        
        for topic_id in opinions_df['matched_topic_id'].unique():
            topic_opinions = opinions_df[
                opinions_df['matched_topic_id'] == "topic_id"
            ]
            
            position_counts = topic_opinions['predicted_type'].value_counts()
            
            samples = {
                'claims': topic_opinions[topic_opinions['predicted_type'] == 'claim']['text'].head(3).tolist(),
                'evidence': topic_opinions[topic_opinions['predicted_type'] == 'evidence']['text'].head(3).tolist(),
                'counterclaims': topic_opinions[topic_opinions['predicted_type'] == 'counterclaim']['text'].head(2).tolist(),
                'rebuttals': topic_opinions[topic_opinions['predicted_type'] == 'rebuttal']['text'].head(2).tolist(),
            }
            
            topic_analysis[topic_id] = {
                'total_opinions': len(topic_opinions),
                'position_distribution': position_counts.to_dict(),
                'position_percentages': (position_counts / len(topic_opinions) * 100).to_dict(),
                'samples': samples,
                'avg_confidence': topic_opinions['type_confidence'].mean()
            }
        
        logger.info(f"Analyzed {len(topic_analysis)} topics")
        return topic_analysis
    
    def create_prompt(self, topic_text, analysis):     
        total = analysis['total_opinions']
        dist = analysis['position_distribution']
        pct = analysis['position_percentages']
        samples = analysis['samples']
        
        prompt = f"""
            Based on social media analysis of {total} opinions about the following topic:

            TOPIC:
            {topic_text}

            ANALYSIS RESULTS:
            - Claims supporting the topic: {dist.get('claim', 0)} opinions ({pct.get('claim', 0):.1f}%)
            - Supporting Evidence: {dist.get('evidence', 0)} opinions ({pct.get('evidence', 0):.1f}%)
            - Counter-claims opposing: {dist.get('counterclaim', 0)} opinions ({pct.get('counterclaim', 0):.1f}%)
            - Rebuttals/Counter-arguments: {dist.get('rebuttal', 0)} opinions ({pct.get('rebuttal', 0):.1f}%)

            SAMPLE OPINIONS:
            Supporting Claims:
            {self._format_samples(samples.get('claims', []))}

            Evidence Provided:
            {self._format_samples(samples.get('evidence', []))}

            Counter-claims:
            {self._format_samples(samples.get('counterclaims', []))}

            Rebuttals:
            {self._format_samples(samples.get('rebuttals', []))}

            Please generate a concise, professional conclusion (2-3 sentences) that:
            1. Summarizes the overall sentiment and distribution of opinions
            2. Highlights the key supporting evidence
            3. Acknowledges opposing viewpoints
            4. Provides insight into the balance of arguments

            CONCLUSION:"""
        
        return prompt
    
    def _format_samples(self, samples):
        if not samples:
            return "No opinions in this category"
        return "\n".join([f"  - {s[:150]}..." if len(s) > 150 else f"  - {s}" for s in samples])
    
    def generate_conclusion_huggingface(self,opinions_texts,max_length= 150,min_length = 50):
        combined_text = " ".join(opinions_texts[:10])
        summary = self.summarizer(combined_text, max_length=max_length, min_length=min_length)        
        return summary[0]['summary_text']
    
    def generate_conclusions(self, opinions_df, topics_df, output_path = "data/processed/generated_conclusions.csv"):
        logger.info("Generating conclusions for all topics...")
        topic_analysis = self.analyze_opinions_by_topic(opinions_df)
        conclusions = []
        for topic_id, topic_text in zip(topics_df['topic_id'], topics_df['text']):
            logger.info(f"Processing topic: {topic_id}")

            if topic_id not in topic_analysis:
                logger.warning(f"No opinions found for topic {topic_id}")
                conclusions.append({
                    'topic_id': topic_id,
                    'topic_text': topic_text,
                    'generated_conclusion': 'No sufficient data for conclusion',
                    'opinion_count': 0
                })
                continue
            
            analysis = topic_analysis[topic_id]
            
            try:
                prompt = self.create_prompt(topic_text, analysis)
                
                opinions_sample = opinions_df[
                    opinions_df['topic_id'] == topic_id
                ]['text'].tolist()
                conclusion_text = self.generate_conclusion_huggingface(opinions_sample)
                
                conclusions.append({
                    'topic_id': topic_id,
                    'topic_text': topic_text,
                    'generated_conclusion': conclusion_text,
                    'opinion_count': analysis['total_opinions'],
                    'claim_count': analysis['position_distribution'].get('claim', 0),
                    'evidence_count': analysis['position_distribution'].get('evidence', 0),
                    'counterclaim_count': analysis['position_distribution'].get('counterclaim', 0),
                    'rebuttal_count': analysis['position_distribution'].get('rebuttal', 0),
                    'avg_confidence': analysis['avg_confidence']
                })
                
            except Exception as e:
                logger.error(f"Error generating conclusion for {topic_id}: {e}")
                conclusions.append({
                    'topic_id': topic_id,
                    'topic_text': topic_text,
                    'generated_conclusion': f'Error: {str(e)}',
                    'opinion_count': analysis['total_opinions']
                })
        
        conclusions_df = pd.DataFrame(conclusions)
        
        if output_path:
            conclusions_df.to_csv(output_path, index=False)
            logger.info(f"Conclusions saved to {output_path}")
        
        return conclusions_df
    

if __name__ == "__main__":
    generator = ConclusionGenerator(config_path="configs/config.yaml")

    opinions_df = pd.DataFrame(pd.read_csv("data/processed/opinions_with_matches.csv"))
    topics_df = pd.DataFrame(pd.read_csv("data/topics_clean.csv"))

    a = generator.generate_conclusions(
        opinions_df=opinions_df,
        topics_df=topics_df)
