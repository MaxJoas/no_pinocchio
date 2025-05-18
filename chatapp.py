import gradio as gr
import numpy as np
import logging
import requests
import json
import time
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the required packages, with fallbacks if not available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers package not found. NLI will not be available.")
    TRANSFORMERS_AVAILABLE = False

class BSDetector:
    """
    Implementation of BSDetector algorithm with improved Ollama handling
    """
    
    def __init__(self, 
                 llm_name="qwen2.5:0.5b",
                 use_nli=True,
                 nli_model_name="cross-encoder/nli-deberta-v3-small",
                 alpha=0.7,  
                 beta=0.5,   
                 num_samples=3,
                 temperature=0.8):
        """Initialize the BSDetector with specified parameters"""
        self.llm_name = llm_name
        self.api_base = "http://localhost:11434/api"
        
        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.num_samples = num_samples
        self.temperature = temperature
        
        # Initialize NLI pipeline if available
        self.use_nli = use_nli and TRANSFORMERS_AVAILABLE
        self.nli_pipeline = None
        
        if self.use_nli:
            try:
                logger.info(f"Loading NLI pipeline with model: {nli_model_name}")
                self.nli_pipeline = pipeline("zero-shot-classification", 
                                            model=nli_model_name)
                logger.info("NLI pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load NLI pipeline: {e}")
                logger.warning("Falling back to simpler similarity metrics")
                self.use_nli = False
    
    def _generate_response(self, prompt, temperature=0.0):
        """Generate a response with robust error handling"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.llm_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            # Make the API call
            response = requests.post(
                f"{self.api_base}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                raise Exception(f"API returned error status: {response.status_code}, {response.text}")
            
            # Safely parse the JSON response
            try:
                # Extract the first line to avoid JSON parsing issues
                json_str = response.text.split('\n')[0].strip()
                result = json.loads(json_str)
                return result.get("response", "")
            except json.JSONDecodeError:
                try:
                    # Try using response.json() method
                    result = response.json()
                    return result.get("response", "")
                except Exception:
                    # Manual extraction if all else fails
                    if '"response":"' in response.text:
                        start = response.text.find('"response":"') + 12
                        end = response.text.find('"', start)
                        if end > start:
                            return response.text[start:end]
            
            logger.error("Failed to parse Ollama API response")
            return ""
            
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: Failed to generate response. Please check if Ollama is running with model {self.llm_name}."
    
    def _nli_contradiction_score(self, reference: str, generated: str) -> float:
        """Calculate contradiction score using NLI pipeline"""
        if not self.use_nli or self.nli_pipeline is None:
            return self._fallback_similarity(reference, generated)
            
        try:
            # Define contradiction and entailment as classes
            labels = ["contradiction", "entailment"]
            
            # Check both directions
            result1 = self.nli_pipeline(reference, 
                                       candidate_labels=labels,
                                       hypothesis=generated)
            
            result2 = self.nli_pipeline(generated, 
                                       candidate_labels=labels,
                                       hypothesis=reference)
            
            # Extract contradiction probabilities
            contradiction_idx1 = result1['labels'].index('contradiction')
            contradiction_prob1 = result1['scores'][contradiction_idx1]
            
            contradiction_idx2 = result2['labels'].index('contradiction')
            contradiction_prob2 = result2['scores'][contradiction_idx2]
            
            # Average the contradiction probabilities
            avg_contradiction_prob = (contradiction_prob1 + contradiction_prob2) / 2
            
            return 1.0 - avg_contradiction_prob
            
        except Exception as e:
            logger.warning(f"Error in NLI processing: {e}")
            return self._fallback_similarity(reference, generated)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Calculate fallback similarity using Jaccard similarity"""
        def preprocess(text):
            text = text.lower().strip()
            words = ''.join(c if c.isalnum() else ' ' for c in text).split()
            return words
            
        words1 = preprocess(text1)
        words2 = preprocess(text2)
        
        if not words1 or not words2:
            return 0.0
            
        set1 = set(words1)
        set2 = set(words2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        similarity = intersection / union
        return similarity
    
    def _exact_match(self, text1: str, text2: str) -> float:
        """Calculate exact match (indicator function r_i in paper)"""
        return 1.0 if text1.strip() == text2.strip() else 0.0
    
    def _compute_similarity(self, reference: str, generated: str) -> float:
        """Compute combined similarity score: o_i = α*s_i + (1-α)*r_i"""
        # Get similarity score (s_i in paper)
        if self.use_nli and self.nli_pipeline is not None:
            sim = self._nli_contradiction_score(reference, generated)
        else:
            sim = self._fallback_similarity(reference, generated)
        
        # Get exact match indicator (r_i in paper)
        exact_match = self._exact_match(reference, generated)
        
        # Combine with alpha weight (o_i in paper)
        similarity = self.alpha * sim + (1 - self.alpha) * exact_match
        
        return similarity
    
    def observed_consistency(self, question: str, reference_answer: str) -> float:
        """Measure observed consistency: O = (1/k) * Σo_i"""
        # Template for Chain-of-Thought prompting
        cot_template = f"""Please strictly use the following template to provide answer:
explanation: [insert step-by-step analysis], answer: [provide your answer]

Question: {question}"""
        
        similarities = []
        for i in range(self.num_samples):
            try:
                logger.info(f"Generating alternative answer {i+1}/{self.num_samples}")
                alternative_answer = self._generate_response(cot_template, temperature=self.temperature)
                
                # Extract the final answer
                if "answer:" in alternative_answer.lower():
                    answer_part = alternative_answer.lower().split("answer:")[1].strip()
                else:
                    answer_part = alternative_answer
                
                # Compute similarity
                similarity = self._compute_similarity(reference_answer, answer_part)
                similarities.append(similarity)
                logger.info(f"Similarity for alternative {i+1}: {similarity:.3f}")
                
            except Exception as e:
                logger.error(f"Error in observed consistency: {e}")
                similarities.append(0.5)
        
        # Return average similarity as observed consistency (O in paper)
        return np.mean(similarities) if similarities else 0.5
    
    def self_reflection_certainty(self, question: str, reference_answer: str) -> float:
        """Calculate self-reflection certainty: S = (score_1 + score_2) / 2"""
        # First reflection prompt
        reflection_prompt1 = f"""Question: {question}, Proposed Answer: {reference_answer}
Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
        
        # Second reflection prompt
        reflection_prompt2 = f"""Question: {question}, Proposed Answer: {reference_answer}
Are you really sure the proposed answer is correct?
Choose again: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
        
        try:
            # Get responses
            logger.info("Generating self-reflection response 1")
            reflection1 = self._generate_response(reflection_prompt1)
            
            logger.info("Generating self-reflection response 2")
            reflection2 = self._generate_response(reflection_prompt2)
            
            # Extract answers
            def extract_letter(reflection):
                if "answer:" in reflection.lower():
                    letter = reflection.lower().split("answer:")[1].strip()[0]
                    if letter in "abc":
                        return letter
                for letter in "abc":
                    if f"({letter.upper()})" in reflection:
                        return letter
                return "c"  # Default to "not sure" if parsing fails
            
            letter1 = extract_letter(reflection1)
            letter2 = extract_letter(reflection2)
            
            logger.info(f"Self-reflection result 1: {letter1}")
            logger.info(f"Self-reflection result 2: {letter2}")
            
            # Convert to numerical values: A=1.0, B=0.0, C=0.5
            values = {"a": 1.0, "b": 0.0, "c": 0.5}
            score1 = values.get(letter1, 0.5)
            score2 = values.get(letter2, 0.5)
            
            # Return average score
            return (score1 + score2) / 2
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            return 0.5
    
    def compute_confidence(self, question: str, answer: Optional[str] = None) -> Tuple[float, str]:
        """Compute overall confidence: C = β*O + (1-β)*S"""
        # If no answer provided, generate one
        if answer is None:
            logger.info("Generating initial answer")
            answer = self._generate_response(question)
            logger.info(f"Generated answer: {answer}")
        
        # Compute the two components
        logger.info("Computing observed consistency")
        consistency = self.observed_consistency(question, answer)
        
        logger.info("Computing self-reflection certainty")
        reflection = self.self_reflection_certainty(question, answer)
        
        logger.info(f"Observed consistency: {consistency:.3f}")
        logger.info(f"Self-reflection certainty: {reflection:.3f}")
        
        # Combine with beta weighting
        confidence = self.beta * consistency + (1 - self.beta) * reflection
        
        return confidence, answer

    def change_model(self, model_name):
        """Change the model being used"""
        self.llm_name = model_name
        logger.info(f"Changed model to {model_name}")
        return f"Model changed to {model_name}"

    def process_message(self, message: str, get_alternatives: bool = False, num_candidates: int = 3) -> Dict:
        """Process a message and return response with confidence score"""
        # Set num_candidates
        try:
            num_candidates = int(num_candidates)
            if num_candidates < 1:
                num_candidates = 1
            elif num_candidates > 5:
                num_candidates = 5
        except:
            num_candidates = 3
            
        # Process the message
        if get_alternatives:
            logger.info(f"Processing message with alternatives (candidates={num_candidates})")
            
            # Generate multiple answers and pick best
            all_candidates = []
            best_answer = None
            best_confidence = -1
            
            for i in range(num_candidates):
                try:
                    logger.info(f"Generating candidate {i+1}/{num_candidates}")
                    answer = self._generate_response(message, temperature=self.temperature)
                    confidence, _ = self.compute_confidence(message, answer)
                    
                    all_candidates.append({
                        "answer": answer,
                        "confidence": confidence,
                        "rank": i+1
                    })
                    
                    # Track best answer
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_answer = answer
                        
                    logger.info(f"Candidate {i+1} confidence: {confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error generating candidate {i+1}: {e}")
            
            # Sort candidates by confidence
            sorted_candidates = sorted(all_candidates, key=lambda x: x["confidence"], reverse=True)
            
            # Update ranks
            for i, candidate in enumerate(sorted_candidates):
                candidate["rank"] = i + 1
            
            # Format response and alternatives
            if best_answer and best_confidence > 0:
                response = f"{best_answer}\n\n[Confidence: {best_confidence:.2f}]"
                alternatives = [
                    f"Rank {c['rank']}: {c['answer'][:100]}... (Confidence: {c['confidence']:.2f})"
                    for c in sorted_candidates
                ]
                return response, alternatives
            else:
                return "Could not generate a confident answer.", []
                
        else:
            logger.info("Processing message for single answer")
            confidence, answer = self.compute_confidence(message)
            response = f"{answer}\n\n[Confidence: {confidence:.2f}]"
            return response, []

# Initialize the detector
bs_detector = BSDetector(
    llm_name="qwen2.5:0.5b",
    use_nli=TRANSFORMERS_AVAILABLE,
    num_samples=2,  # Reduced for speed
    temperature=0.8
)

# Create a simple chat app with blocks (no State issues)
with gr.Blocks(title="BSDetector Chat") as demo:
    gr.Markdown("# BSDetector Chat")
    gr.Markdown("Ask questions and get answers with confidence scores")
    
    # Chat interface components
    chatbot = gr.Chatbot(
        type="messages", # Modern messages format avoiding warning
        height=500,
        label="Conversation",
        show_copy_button=True,
        avatar_images=("👤", "🤖")
    )
    
    # Input and send button in the same row
    with gr.Row():
        message = gr.Textbox(
            placeholder="Type your question here...",
            label="Your question",
            lines=2,
            scale=9
        )
        send_button = gr.Button("Send", scale=1, variant="primary", size="lg")
    
    # Settings row
    with gr.Row():
        # Column for buttons
        with gr.Column(scale=3):
            clear_button = gr.Button("Clear Chat")
            
        # Column for model selection
        with gr.Column(scale=7):
            model_dropdown = gr.Dropdown(
                ["qwen2.5:0.5b", "phi", "llama2", "mistral", "gemma:2b"],
                label="LLM Model",
                value="qwen2.5:0.5b"
            )
    
    # Alternatives and settings
    with gr.Accordion("Alternative Answers", open=False):
        with gr.Row():
            generate_alternatives = gr.Checkbox(
                label="Generate multiple answers and pick best",
                value=False
            )
            num_candidates = gr.Slider(
                minimum=2, 
                maximum=5, 
                value=3, 
                step=1, 
                label="Number of candidates"
            )
        
        alternatives_box = gr.Textbox(
            label="Alternatives",
            value="No alternatives generated yet.",
            lines=8,
            interactive=False
        )
    
    # Function to handle sending a message
    def on_message(message, chat_history, model, gen_alts, num_cands):
        # Skip if message is empty
        if not message.strip():
            return message, chat_history, alternatives_box.value
        
        # Update model if needed
        if model != bs_detector.llm_name:
            bs_detector.change_model(model)
        
        # Process the message
        try:
            # Add thinking message first
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "Thinking..."})
            yield "", chat_history, alternatives_box.value
            
            # Process the message
            response, alternatives = bs_detector.process_message(
                message, 
                get_alternatives=gen_alts,
                num_candidates=int(num_cands)
            )
            
            # Update alternatives display
            alternatives_text = "\n\n".join(alternatives) if alternatives else "No alternatives generated."
            
            # Update the assistant message (replace the "Thinking..." placeholder)
            chat_history[-1] = {"role": "assistant", "content": response}
            
            yield "", chat_history, alternatives_text
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Replace the "Thinking..." message with the error
            chat_history[-1] = {"role": "assistant", "content": error_msg}
            yield "", chat_history, "Error processing alternatives."
    
    def clear_chat():
        return [], "No alternatives generated yet."
    
    # Connect the components
    message.submit(
        on_message,
        [message, chatbot, model_dropdown, generate_alternatives, num_candidates],
        [message, chatbot, alternatives_box]
    )
    
    # Connect the send button to the same function
    send_button.click(
        on_message,
        [message, chatbot, model_dropdown, generate_alternatives, num_candidates],
        [message, chatbot, alternatives_box]
    )
    
    # Connect the clear button
    clear_button.click(clear_chat, outputs=[chatbot, alternatives_box])
    
    # Add informative section
    with gr.Accordion("About BSDetector", open=False):
        gr.Markdown("""
        BSDetector quantifies uncertainty in language model responses using:
        
        1. **Observed Consistency**: Checking if alternative responses agree
        2. **Self-Reflection**: Having the model assess its own confidence
        
        The algorithm combines these into a single confidence score that helps you
        know when to trust or be skeptical of the model's responses.
        
        Confidence scores range from 0 (very uncertain) to 1 (very confident).
        
        **Technical note**: Make sure Ollama is running with the selected model:
        ```
        ollama run qwen2.5:0.5b
        ```
        """)

# Launch the app
if __name__ == "__main__":
    print("Starting BSDetector Gradio interface...")
    print("Make sure Ollama is running with your model (e.g., 'ollama run qwen2.5:0.5b')")
    
    demo.launch(share=False)