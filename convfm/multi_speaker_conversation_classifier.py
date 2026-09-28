#!/usr/bin/env python3
"""
Streaming Multi-Speaker Conversation Classifier
Reads transcripts from multiple speakers and classifies them into conversations using Qwen2.5-0.5B-Instruct
with conversation memory and streaming context for consistent classification across time batches.
"""

import torch
import time
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import random
import string
import json
import re
from collections import defaultdict
import os
import openai
import google.generativeai as genai
import anthropic
import requests
import csv

class MultiSpeakerClassifier:
    def __init__(self, model_name, conversation_groupings=None):
        self.model_name = model_name
        # Simple OpenAI model detection: any model name containing 'gpt' or starting with 'o' is OpenAI
        self.is_openai = 'gpt' in model_name.lower() or model_name.lower().startswith('o')
        # Simple Gemini model detection: any model name containing 'gemini' is Gemini
        self.is_gemini = 'gemini' in model_name.lower()
        # Simple Claude model detection: any model name containing 'claude' is Claude
        self.is_claude = 'claude' in model_name.lower()
        # Simple Together model detection: any model name containing 'together', 'togethercomputer', or 'llama' is Together
        self.is_together = (
            'together' in model_name.lower() or
            'togethercomputer' in model_name.lower() or
            'llama' in model_name.lower()
        )
        self.openai_model = model_name.replace('openai/', '') if model_name.startswith('openai/') else model_name if self.is_openai else None
        self.gemini_model = model_name if self.is_gemini else None
        self.claude_model = model_name if self.is_claude else None
        self.together_model = model_name if self.is_together else None
        self.model = None
        self.tokenizer = None
        self.speaker_files = {}
        self.speaker_positions = {}
        self.seconds_per_batch = 30
        self.conversation_history = []
        self.chat_history = []
        self.max_history_length = 15
        self.conversation_groupings = conversation_groupings
        self.together_api_key = ""
        self.openai_api_key = ""
        self.gemini_api_key = ""
        self.claude_api_key = ""
        # Only load HuggingFace model if not OpenAI, Gemini, Claude, or Together
        if not self.is_openai and not self.is_gemini and not self.is_claude and not self.is_together:
            self.load_model()

    def load_model(self, verbose=True):
        """Load HuggingFace model and tokenizer only if not OpenAI or Gemini."""
        if self.is_openai or self.is_gemini or self.is_claude or self.is_together:
            return  # Never load HuggingFace model for OpenAI or Gemini
        if verbose:
            print(f"Loading {self.model_name} for streaming conversation classification...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if verbose:
                print("✓ Tokenizer loaded successfully")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if verbose:
                print("✓ Model loaded successfully")
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            if verbose:
                print(f"✅ {self.model_name} ready for streaming conversation classification!")
                print(f"🧠 Model supports conversation memory across batches")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def setup_speaker_files(self, speaker_files: Dict[str, str], verbose=True):
        """Setup speaker files and initialize reading positions"""
        self.speaker_files = {}
        self.speaker_positions = {}

        for speaker_id, file_path in speaker_files.items():
            path = Path(file_path)
            if not path.exists():
                if verbose:
                    print(f"⚠️  File not found: {file_path}")
                continue

            # Read all lines from file, try utf-8 then fallback to cp1252
            try:
                f = open(path, 'r', encoding='utf-8')
                lines = [line.strip() for line in f.readlines()]
                f.close()
            except UnicodeDecodeError:
                if verbose:
                    print(f"⚠️  Could not decode {file_path} as utf-8, trying cp1252...")
                f = open(path, 'r', encoding='cp1252')
                lines = [line.strip() for line in f.readlines()]
                f.close()

            self.speaker_files[speaker_id] = lines
            self.speaker_positions[speaker_id] = 0
            if verbose:
                print(f"✓ Loaded {len(lines)} seconds of transcript for {speaker_id}")

    def read_next_batch(self) -> Dict[str, List[str]]:
        """Read next 10 seconds from each speaker"""
        batch = {}

        for speaker_id in self.speaker_files:
            start_pos = self.speaker_positions[speaker_id]
            end_pos = start_pos + self.seconds_per_batch

            # Get the next 10 seconds (lines) for this speaker
            speaker_lines = self.speaker_files[speaker_id][start_pos:end_pos]

            # If we have fewer than 10 lines, pad with [silence]
            while len(speaker_lines) < self.seconds_per_batch and (start_pos + len(speaker_lines)) < len(self.speaker_files[speaker_id]):
                speaker_lines.append("[silence]")

            batch[speaker_id] = speaker_lines

            # Update position for next batch
            self.speaker_positions[speaker_id] = end_pos

        return batch

    def has_more_content(self) -> bool:
        """Stop as soon as any speaker's transcript is exhausted (stop early)."""
        for speaker_id in self.speaker_files:
            if self.speaker_positions[speaker_id] >= len(self.speaker_files[speaker_id]):
                return False
        return True

    def format_batch_for_llm(self, batch: Dict[str, List[str]], batch_number: int, llm_detects_conv_structure: bool) -> str:
        """Format speaker batch for LLM analysis with conversation history and dynamic conversation structure"""
        speakers = list(batch.keys())
        # Find which speakers are in which conversation
        groupings = self.conversation_groupings
        num_convs = len(groupings)
        prompt_parts = []
        prompt_parts.append(f"BATCH {batch_number}")
        prompt_parts.append("")
        prompt_parts.append(f"Available speakers: {', '.join(speakers)}")
        prompt_parts.append("")
        prompt_parts.append("INPUT DATA:")
        prompt_parts.append("Each [silence] token represents 2-4 words of text. Use this to align conversation timing.")
        for speaker_id, lines in batch.items():
            transcript = " ".join(lines)
            prompt_parts.append(f"{speaker_id}: {transcript}")
        prompt_parts.append("")
        # Add a clear separator
        prompt_parts.append("--- END OF INPUT ---")
        prompt_parts.append("")
        if llm_detects_conv_structure:
            prompt_parts.append("TASK:")
            prompt_parts.append("Group speakers into conversations based on who talks to whom.")
            prompt_parts.append("")
            prompt_parts.append("KEY RULES:")
            prompt_parts.append("- [silence] separates different conversations")
            prompt_parts.append("- Speakers who respond to each other are in same conversation")
            prompt_parts.append("- Each speaker can only be in one conversation")
            prompt_parts.append("- Look for turn-taking patterns and direct responses")
            prompt_parts.append("- There are 2-4 conversations; the exact number is unknown")
            prompt_parts.append("- Each conversation has at least two speakers")
            prompt_parts.append("- Try not to group all speakers in one conversation")
            prompt_parts.append("- Do not group based on physical adjacency, group based on actual conversation structure")
            prompt_parts.append("")
            prompt_parts.append("ANALYSIS:")
            prompt_parts.append("- Use your history of previous prompts and results to aid your current processing")
            prompt_parts.append("- Determine the conversation structure correctly before filling in the speakers")
            prompt_parts.append("- Use silence alignment: each [silence] token represents 2-4 words of text")
            prompt_parts.append("- Group speakers who talk in sequence without [silence] between them")
            prompt_parts.append("- Speakers with [silence] at the same position are in different conversations")
            prompt_parts.append("- Look for direct responses (questions/answers, agreements/disagreements)")
            prompt_parts.append("- Look for speakers discussing the same topic or subject matter")
            prompt_parts.append("- Identify speakers who reference each other's statements or respond to specific points")
            prompt_parts.append("- Group speakers who ask and answer questions about the same topic")
            prompt_parts.append("- Consider context: speakers discussing work-related topics are likely in a work conversation")
            prompt_parts.append("- Look for agreement/disagreement patterns between speakers on the same subject")
            prompt_parts.append("- Use topic continuity to confirm conversation groupings identified through timing")
            prompt_parts.append("- Count conversations and ensure reasonable grouping")
            prompt_parts.append("")
            prompt_parts.append("OUTPUT: JSON with conversation groups")
            prompt_parts.append("- Make sure to output in JSON format only, no other text should be outputted")
            prompt_parts.append("- Do not repeat or quote the input text in your response")
            prompt_parts.append("")
            prompt_parts.append("Example JSON format:")
            prompt_parts.append("{")
            prompt_parts.append('  "conversation_i": {"speakers": ["Speaker_XXXXXX", "Speaker_YYYYYY", ...], "topic": "[topic_i]"},')
            prompt_parts.append('  "conversation_j": {"speakers": ...')
            prompt_parts.append('  ...')
            prompt_parts.append("}")
            prompt_parts.append("")
            prompt_parts.append("Now analyze the input and output your JSON answer.")
            return "\n".join(prompt_parts)
                # Original prompt logic below
        prompt_parts.append(f"TASK:")
        prompt_parts.append(f"Group speakers into exactly {num_convs} conversations based on who talks to whom.")
        prompt_parts.append("")
        prompt_parts.append("KEY RULES:")
        prompt_parts.append("- [silence] separates different conversations")
        prompt_parts.append("- Speakers who respond to each other are in same conversation")
        prompt_parts.append("- Each speaker can only be in one conversation")
        prompt_parts.append("- Look for turn-taking patterns and direct responses")
        prompt_parts.append("- Focus on conversation structure, not topics")
        prompt_parts.append("- Do not group based on physical adjacency, group based on actual conversation structure")
        prompt_parts.append("")
        prompt_parts.append("REQUIREMENTS:")
        for i, group in enumerate(groupings):
            prompt_parts.append(f"- Conversation {i+1}: exactly {len(group)} speakers")
        prompt_parts.append(f"- Total: {num_convs} conversations, all {len(speakers)} speakers used exactly once")
        prompt_parts.append("")
        prompt_parts.append("ANALYSIS:")
        prompt_parts.append("- Use your history of previous prompts and results to aid your current processing")
        prompt_parts.append("- Determine the conversation structure correctly before filling in the speakers")
        prompt_parts.append("- Use silence alignment: each [silence] token represents 2-4 words of text")
        prompt_parts.append("- Group speakers who talk in sequence without [silence] between them")
        prompt_parts.append("- Speakers with [silence] at the same position are in different conversations")
        prompt_parts.append("- Look for direct responses (questions/answers, agreements/disagreements)")
        prompt_parts.append("- Look for speakers discussing the same topic or subject matter")
        prompt_parts.append("- Identify speakers who reference each other's statements or respond to specific points")
        prompt_parts.append("- Group speakers who ask and answer questions about the same topic")
        prompt_parts.append("- Consider context: speakers discussing work-related topics are likely in a work conversation")
        prompt_parts.append("- Look for agreement/disagreement patterns between speakers on the same subject")
        prompt_parts.append("- Use topic continuity to confirm conversation groupings identified through timing")
        prompt_parts.append(f"- Ensure exactly {num_convs} conversations with correct speaker counts")
        prompt_parts.append("- Balance speaker distribution across conversations")
        prompt_parts.append("")
        prompt_parts.append("OUTPUT: JSON with conversation groups")
        prompt_parts.append("- Make sure to output in JSON format only, no other text should be outputted")
        prompt_parts.append("- Do not repeat or quote the input text in your response")
        prompt_parts.append("")
        prompt_parts.append("Complete this template with actual speaker names and topics:")
        prompt_parts.append("{")
        num_speakers = len(speakers)
        letter_idx = 0
        for i, group in enumerate(groupings):
            group_size = len(group)
            # Use repeated letters for template speakers (Z, Y, X, etc.)
            group_labels = [f'Speaker_{chr(ord("Z") - letter_idx - j) * 6}' for j in range(group_size)]
            speaker_list = ', '.join([f'"{label}"' for label in group_labels])
            prompt_parts.append(f'  "conversation_{i+1}": {{"speakers": [{speaker_list}], "topic": "REPLACE_WITH_ACTUAL_TOPIC"}}' + (',' if i < num_convs-1 else ''))
            letter_idx += group_size
        prompt_parts.append("}")
        prompt_parts.append("")
        prompt_parts.append("Analyze the transcript and output your JSON:")
        return "\n".join(prompt_parts)

    def classify_conversations_async(self, batch: dict, batch_number: int, llm_detects_conv_structure: bool, verbose=True, trial_number=None):
        """Classify conversations using OpenAI API or HuggingFace model."""
        def classification_worker():
            try:
                prompt_text = self.format_batch_for_llm(batch, batch_number, llm_detects_conv_structure)
                # Do NOT print the LLM prompt here anymore
                classification = ""
                response_time = 0.0
                if self.is_openai:
                    import re
                    client = openai.OpenAI(api_key=self.openai_api_key or os.getenv("OPENAI_API_KEY"))
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            start_time = time.time()
                            # Use max_completion_tokens for o3/o4 models, max_tokens for others
                            token_param = "max_completion_tokens" if self.openai_model.startswith(('o3', 'o4')) else "max_tokens"
                            
                            # Build parameters dict, excluding temperature for o3/o4 models
                            params = {
                                "model": self.openai_model,
                                "messages": [
                                    {"role": "system", "content": "You are a conversation classifier. Complete the template with actual speaker names and topics based on the conversation content."},
                                    {"role": "user", "content": prompt_text}
                                ]
                            }
                            # Use higher token limit for o3/o4 models since they need more tokens
                            token_limit = 2000 if self.openai_model.startswith(('o3', 'o4')) else 200
                            params[token_param] = token_limit
                            
                            # Only add temperature for non-o3/o4 models
                            if not self.openai_model.startswith(('o3', 'o4')):
                                params["temperature"] = 0.0
                            
                            response = client.chat.completions.create(**params)
                            end_time = time.time()
                            response_time = end_time - start_time
                            classification = response.choices[0].message.content
                            
                            # Debug: Check if response is empty
                            if not classification or classification.strip() == "":
                                print(f"⚠️ Empty response from {self.openai_model}")
                                print(f"Response object: {response}")
                                classification = "{}"  # Fallback empty JSON
                            break  # Success, exit retry loop
                        except Exception as e:
                            msg = str(e)
                            if 'rate limit' in msg.lower() or '429' in msg:
                                match = re.search(r"try again in (\d+)s", msg)
                                wait_time = int(match.group(1)) if match else 20
                                print(f"⚠️ Rate limit hit. Sleeping for {wait_time} seconds before retrying batch {batch_number}...")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"\n❌ Error in batch {batch_number} classification: {e}")
                                print("Continuing with next batch...\n")
                                return
                    else:
                        print(f"❌ Error in batch {batch_number} classification: Rate limit exceeded after {max_retries} retries.")
                        print("Continuing with next batch...\n")
                        return
                elif self.is_gemini:
                    import re
                    genai.configure(api_key=self.gemini_api_key or os.getenv("GEMINI_API_KEY"))
                    model = genai.GenerativeModel(self.gemini_model)
                    start_time = time.time()
                    response = model.generate_content(prompt_text)
                    end_time = time.time()
                    response_time = end_time - start_time
                    classification = response.text
                elif self.is_claude:
                    client = anthropic.Anthropic(api_key=self.claude_api_key or os.getenv("ANTHROPIC_API_KEY"))
                    start_time = time.time()
                    response = client.messages.create(
                        model=self.claude_model,
                        max_tokens=200,
                        temperature=0.0,
                        messages=[
                            {"role": "user", "content": prompt_text}
                        ]
                    )
                    end_time = time.time()
                    response_time = end_time - start_time
                    classification = "".join([block.text for block in response.content])
                elif self.is_together:
                    url = "https://api.together.xyz/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.together_api_key or os.getenv('TOGETHER_API_KEY')}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": self.together_model,
                        "messages": [
                            {"role": "system", "content": "You are a conversation classifier. Complete the template with actual speaker names and topics based on the conversation content."},
                            {"role": "user", "content": prompt_text}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.0
                    }
                    start_time = time.time()
                    response = requests.post(url, headers=headers, json=data)
                    end_time = time.time()
                    response_time = end_time - start_time
                    result = response.json()
                    if 'choices' in result and result['choices']:
                        classification = result['choices'][0]['message']['content']
                    else:
                        print("Together API error or unexpected response:", result)
                        classification = ""
                else:
                    prompt_text = self.format_batch_for_llm(batch, batch_number, llm_detects_conv_structure)
                    messages = [
                        {"role": "system", "content": "You are a conversation classifier. You must analyze the speakers and group them into conversations. Output your analysis as JSON with conversation groups."},
                        {"role": "user", "content": prompt_text}
                    ]
                    try:
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                    except:
                        formatted_prompt = prompt_text
                    
                    model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True, truncation=True)
                    if torch.cuda.is_available():
                        model_inputs = model_inputs.to(self.model.device)
                    input_token_count = model_inputs.input_ids.shape[1]
                    start_time = time.time()
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            max_new_tokens=200
                        )
                    end_time = time.time()
                    response_time = end_time - start_time
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    classification = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                                        # Extract only JSON from the response (in case model thinks out loud)
                    import re
                    
                    # Simple JSON extraction - look for {...} pattern
                    json_match = re.search(r'\{.*?\}', classification, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(0)
                        # Check if the extracted JSON is empty or just contains empty braces
                        if extracted_json.strip() in ['{}', '{"}'] or len(extracted_json.strip()) <= 2:
                            classification = ""  # Mark as invalid response
                        else:
                            classification = extracted_json
                    else:
                        classification = ""  # No JSON found, mark as invalid response
                # Print speaker transcripts and LLM output for this batch
                if verbose:
                    header_trial = f"{trial_number}" if trial_number is not None else "?"
                    print(f"\n================ TRIAL {header_trial} BATCH {batch_number} ================")
                    print(f"\n📝 SPEAKER TRANSCRIPTS (Batch {batch_number}):")
                    for speaker_id, lines in batch.items():
                        transcript = " ".join(lines)
                        print(f"   {speaker_id}: {transcript}")
                    print("\n🤖 LLM OUTPUT:")
                    print(classification)
                if not verbose:
                    print(f"Batch {batch_number} done")
                self.conversation_history.append({
                    'batch_number': batch_number,
                    'speakers': batch,
                    'classification': classification,
                    'response_time': response_time,
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"\n❌ Error in batch {batch_number} classification: {e}")
                print("Continuing with next batch...\n")
        thread = threading.Thread(target=classification_worker)
        thread.daemon = True
        thread.start()
        return thread

    def _parse_response(self, response: str) -> str:
        """Parse Qwen2.5-0.5B-Instruct response, handling any special formatting"""
        # Remove thinking content if present (though less common in 0.5B model)
        if "<think>" in response and "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                actual_response = parts[1].strip()
            else:
                actual_response = response
        else:
            actual_response = response

        return actual_response.strip()

    def _display_results(self, batch: Dict[str, List[str]], batch_number: int, classification: str, response_time: float):
        """Display classification results with timing information"""
        print(f"\n{'='*80}")
        print(f"🎯 STREAMING CONVERSATION CLASSIFICATION - BATCH {batch_number}")
        print(f"{'='*80}")

        print("\n📝 SPEAKER TRANSCRIPTS:")
        for speaker_id, lines in batch.items():
            transcript = " ".join(lines)
            print(f"   {speaker_id}: {transcript}")

        print(f"\n🤖 QWEN2.5-0.5B-INSTRUCT CLASSIFICATION:")
        print(classification)

        print(f"\n⏱️  Response Time: {response_time:.2f} seconds")

        print(f"\n{'='*80}\n")

def run_classification_test(
    model_name: str,
    trials: int = 3,
    num_2person_convos: int = 1,
    num_3person_convos: int = 1,
    num_4person_convos: int = 0,
    llm_detects_conv_structure: bool = False,
    use_dummy_transcripts: bool = False,
    keep_history: bool = False,
    verbose: bool = True,
    debug: bool = False
):
    """
    Run a multi-speaker conversation classification test with specified parameters.

    Args:
        model_name: Name of the LLM model to use
        trials: Number of trials to run
        num_2person_convos: Number of 2-person conversations to include
        num_3person_convos: Number of 3-person conversations to include
        num_4person_convos: Number of 4-person conversations to include
        llm_detects_conv_structure: If True, LLM must infer conversation structure
        use_dummy_transcripts: If True, use dummy transcripts from ./dummy_txts
        keep_history: If True, keep one model instance across all trials
        verbose: If True, print detailed output during processing
        debug: If True, output failed test cases with ground truth groupings
    """

    if verbose:
        print("🎙️  Streaming Multi-Speaker Conversation Classifier")
        print(f"💫 Powered by {model_name} with Memory")
        print("=" * 55)

    # Gather all possible transcript file names for each conversation type
    if use_dummy_transcripts:
        base_path = './dummy_txts/dummy'
        max_convos = 10
    else:
        base_path = './transcripts/speaker'
        max_convos = 50
    all_2p = [f'{base_path}0{idx:02d}{spk}_transcript.txt' for idx in range(1, max_convos+1) for spk in 'ab']
    all_3p = [f'{base_path}1{idx:02d}{spk}_transcript.txt' for idx in range(1, max_convos+1) for spk in 'abc']
    all_4p = [f'{base_path}2{idx:02d}{spk}_transcript.txt' for idx in range(1, max_convos+1) for spk in 'abcd']

    used_convo_ids = set()
    cumulative_stats = {
        'total_correct': 0,
        'total_incorrect': 0,
        'total_partially_correct': 0,
        'valid_responses': 0,
        'invalid_responses': 0,
        'total_classifications': 0,
        'total_response_time': 0.0,
        'total_batches': 0,
    }

    # If keeping history, create one classifier instance
    shared_classifier = None
    if keep_history:
        shared_classifier = MultiSpeakerClassifier(model_name)
        shared_classifier.load_model(verbose=verbose)

    # Collect per-trial stats for file output
    all_trial_stats = []

    # Collect failed test cases for debug output
    failed_cases = []
    
    # Create CSV filename
    csv_filename = create_csv_filename(model_name, trials, num_2person_convos, num_3person_convos, 
                                     num_4person_convos, llm_detects_conv_structure, use_dummy_transcripts, 
                                     keep_history, debug)
    
    # Open CSV file for writing
    csvfile = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csvfile)
    
    # Write CSV header with test parameters
    csv_writer.writerow(["CONVERSATION CLASSIFICATION RESULTS"])
    csv_writer.writerow([f"Model: {model_name}"])
    csv_writer.writerow([f"Trials: {trials}"])
    csv_writer.writerow([f"2-person conversations: {num_2person_convos}"])
    csv_writer.writerow([f"3-person conversations: {num_3person_convos}"])
    csv_writer.writerow([f"4-person conversations: {num_4person_convos}"])
    csv_writer.writerow([f"LLM detects conv structure: {llm_detects_conv_structure}"])
    csv_writer.writerow([f"Use dummy transcripts: {use_dummy_transcripts}"])
    csv_writer.writerow([f"Keep history: {keep_history}"])
    csv_writer.writerow([f"Debug: {debug}"])
    csv_writer.writerow([])  # Empty row for spacing

    for trial in range(1, trials+1):
        print(f"\n================= TRIAL {trial} =================")
        print(f"Settings: 2p={num_2person_convos}, 3p={num_3person_convos}, 4p={num_4person_convos}, detect={llm_detects_conv_structure}, dummy={use_dummy_transcripts}, history={keep_history}")

        # Randomly select conversations for this trial, avoiding repeats
        available_2p = [i for i in range(1, max_convos+1) if f'0{i:02d}' not in used_convo_ids]
        available_3p = [i for i in range(1, max_convos+1) if f'1{i:02d}' not in used_convo_ids]
        available_4p = [i for i in range(1, max_convos+1) if f'2{i:02d}' not in used_convo_ids]

        # Check if we have enough conversations available, reset if not
        if (len(available_2p) < num_2person_convos or
            len(available_3p) < num_3person_convos or
            len(available_4p) < num_4person_convos):
            if verbose:
                print(f"⚠️  Resetting conversation pool - not enough conversations available")
                print(f"   Available: 2p={len(available_2p)}, 3p={len(available_3p)}, 4p={len(available_4p)}")
                print(f"   Needed: 2p={num_2person_convos}, 3p={num_3person_convos}, 4p={num_4person_convos}")
            used_convo_ids.clear()
            available_2p = [i for i in range(1, max_convos+1)]
            available_3p = [i for i in range(1, max_convos+1)]
            available_4p = [i for i in range(1, max_convos+1)]

        indices_2p = random.sample(available_2p, num_2person_convos) if num_2person_convos > 0 else []
        indices_3p = random.sample(available_3p, num_3person_convos) if num_3person_convos > 0 else []
        indices_4p = random.sample(available_4p, num_4person_convos) if num_4person_convos > 0 else []
        # Mark these as used
        for idx in indices_2p:
            used_convo_ids.add(f'0{idx:02d}')
        for idx in indices_3p:
            used_convo_ids.add(f'1{idx:02d}')
        for idx in indices_4p:
            used_convo_ids.add(f'2{idx:02d}')

        transcript_files = []
        conversation_groupings = []
        # 2-person
        for idx in indices_2p:
            num_str = f"0{idx:02d}"
            start_idx = len(transcript_files)
            transcript_files.append(f'{base_path}{num_str}a_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}b_transcript.txt')
            conversation_groupings.append([start_idx, start_idx+1])
        # 3-person
        for idx in indices_3p:
            num_str = f"1{idx:02d}"
            start_idx = len(transcript_files)
            transcript_files.append(f'{base_path}{num_str}a_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}b_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}c_transcript.txt')
            conversation_groupings.append([start_idx, start_idx+1, start_idx+2])
        # 4-person
        for idx in indices_4p:
            num_str = f"2{idx:02d}"
            start_idx = len(transcript_files)
            transcript_files.append(f'{base_path}{num_str}a_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}b_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}c_transcript.txt')
            transcript_files.append(f'{base_path}{num_str}d_transcript.txt')
            conversation_groupings.append([start_idx, start_idx+1, start_idx+2, start_idx+3])
        random.shuffle(transcript_files)
        # Generate random 6-digit numbers for speaker labels instead of letters
        SPEAKER_LABELS = [f"Speaker_{random.randint(100000, 999999)}" for i in range(len(transcript_files))]
        # After shuffling, update conversation_groupings to refer to the new indices
        original_files = transcript_files.copy()
        shuffled_files = transcript_files.copy()
        file_to_new_idx = {f: i for i, f in enumerate(shuffled_files)}
        shuffled_conversation_groupings = []
        for group in conversation_groupings:
            shuffled_group = [file_to_new_idx[original_files[idx]] for idx in group]
            shuffled_conversation_groupings.append(shuffled_group)
        # Build ground truth by grouping speakers with the same conversation number in their transcript filename
        GROUND_TRUTH_CONVOS = []
        convnum_to_speakers = defaultdict(set)
        for idx, path in enumerate(transcript_files):
            label = SPEAKER_LABELS[idx]  # Use the same random labels we generated above
            m = re.search(r'(?:speaker|dummy)(\d+)[a-z]_transcript', path)
            if m:
                convnum = m.group(1)
                convnum_to_speakers[convnum].add(label)
        GROUND_TRUTH_CONVOS = list(convnum_to_speakers.values())

        # Print ground truth groupings at the start of the trial
        if verbose:
            print("\n=== GROUND TRUTH CONVERSATION GROUPINGS ===")
            for i, group in enumerate(GROUND_TRUTH_CONVOS):
                print(f"{{ {', '.join(sorted(group))} }}")
            print("==========================================\n")

        # Map transcript files to speaker labels
        if len(transcript_files) != len(SPEAKER_LABELS):
            print(f"❌ transcript_files count ({len(transcript_files)}) does not match SPEAKER_LABELS count ({len(SPEAKER_LABELS)})")
            continue
        speaker_files = {label: path for label, path in zip(SPEAKER_LABELS, transcript_files)}
        if verbose:
            print('\n📁 Using the following transcript files:')
            for label, path in speaker_files.items():
                print(f'  {label}: {path}')

        # Initialize or reuse classifier
        if keep_history:
            classifier = shared_classifier
            # Reset only the per-trial state, keep model and chat history
            classifier.speaker_files = {}
            classifier.speaker_positions = {}
            classifier.conversation_history = []
            classifier.conversation_groupings = shuffled_conversation_groupings
        else:
            classifier = MultiSpeakerClassifier(model_name, conversation_groupings=shuffled_conversation_groupings)
            classifier.load_model(verbose=verbose)
        # Setup speaker files
        classifier.setup_speaker_files(speaker_files, verbose=verbose)
        # Debug: Print loaded speaker files and positions
        if verbose:
            print('Speaker files loaded:')
            for k, v in classifier.speaker_files.items():
                print(f'  {k}: {len(v)} lines')
            print('Speaker positions:', classifier.speaker_positions)
            print('has_more_content:', classifier.has_more_content())
        # Prepare first batch and print LLM prompt ONCE per trial
        batch_number = 1
        active_threads = []
        if verbose:
            first_batch = classifier.read_next_batch()
            first_prompt = classifier.format_batch_for_llm(first_batch, batch_number, llm_detects_conv_structure)
            print(f"\n================ LLM PROMPT (Trial {trial}) ================")
            print(first_prompt)
            print("================ END OF PROMPT ================\n")
            # Reset positions for actual batch processing
            classifier.speaker_positions = {k: 0 for k in classifier.speaker_positions}
        while classifier.has_more_content():
            # Read next batch from all speakers
            batch = classifier.read_next_batch()
            # Check if batch has any content
            has_content = any(
                any(line.strip() and line != "[silence]" for line in lines)
                for lines in batch.values()
            )
            if not has_content:
                if verbose:
                    print(f"⏭️  Batch {batch_number}: No content, skipping...")
                batch_number += 1
                continue
            # Wait for previous thread to complete (to avoid overwhelming the model)
            if active_threads:
                active_threads[-1].join()
            # Start classification for this batch
            thread = classifier.classify_conversations_async(batch, batch_number, llm_detects_conv_structure, verbose=verbose, trial_number=trial)
            active_threads.append(thread)
            batch_number += 1
            # Small delay between batches
            time.sleep(2)
        # Wait for all threads to complete
        for thread in active_threads:
            thread.join()
        # Only print once for the last batch
        if not verbose and batch_number > 1:
            print(f"Batch {batch_number-1} done")
        if verbose:
            print("✅ All batches processed!")

        # After processing, accumulate stats from this trial
        # We'll extract these from the classifier's conversation_history and _display_summary logic
        # (Re-run the evaluation code here to accumulate)
        ground_truth = GROUND_TRUTH_CONVOS
        total_correct = 0
        total_partially_correct = 0
        total_incorrect = 0
        valid_responses = 0
        invalid_responses = 0
        total_classifications = 0
        total_response_time = 0.0
        total_batches = 0

        # Track failed cases for debug output
        trial_failed_cases = []
        for entry in classifier.conversation_history:
            output = entry['classification']
            total_response_time += entry['response_time']
            total_batches += 1
            if verbose:
                print(f"\nBatch {entry['batch_number']} ({entry['response_time']:.2f}s):")
            else:
                print(f"Batch {entry['batch_number']}", end=" ")
            # Find the first {...} block (the JSON)
            match = re.search(r'\{[\s\S]*\}', output)
            if not match:
                if verbose:
                    print("  Output: INVALID (no JSON found)")
                    print("  Correct: 0, Partially correct: 0, Incorrect: 0")
                else:
                    print("INVALID", end=" ")
                invalid_responses += 1
                # Track failed case for debug
                if debug:
                    trial_failed_cases.append({
                        'trial': trial,
                        'batch': entry['batch_number'],
                        'ground_truth': ground_truth,
                        'batch_data': entry['speakers'],
                        'output': output,
                        'error': 'INVALID (no JSON found)',
                        'response_time': entry['response_time'],
                        'test_params': {
                            'llm_detects_conv_structure': llm_detects_conv_structure,
                            'use_dummy_transcripts': use_dummy_transcripts,
                            'keep_history': keep_history,
                            'num_2person_convos': num_2person_convos,
                            'num_3person_convos': num_3person_convos,
                            'num_4person_convos': num_4person_convos
                        }
                    })
                continue
            try:
                output_json = json.loads(match.group(0))
                # Check for duplicate speaker assignments
                all_speakers = []
                for conv in output_json.values():
                    all_speakers.extend(conv.get('speakers', []))
                if len(all_speakers) != len(set(all_speakers)):
                    if verbose:
                        print("  Output: INVALID (duplicate speaker assignment)")
                        print("  Correct: 0, Partially correct: 0, Incorrect: 0")
                    else:
                        print("INVALID", end=" ")
                    invalid_responses += 1
                    # Track failed case for debug
                    if debug:
                        trial_failed_cases.append({
                            'trial': trial,
                            'batch': entry['batch_number'],
                            'ground_truth': ground_truth,
                            'batch_data': entry['speakers'],
                            'output': output,
                            'error': 'INVALID (duplicate speaker assignment)',
                            'response_time': entry['response_time'],
                            'test_params': {
                                'llm_detects_conv_structure': llm_detects_conv_structure,
                                'use_dummy_transcripts': use_dummy_transcripts,
                                'keep_history': keep_history,
                                'num_2person_convos': num_2person_convos,
                                'num_3person_convos': num_3person_convos,
                                'num_4person_convos': num_4person_convos
                            }
                        })
                    continue
                valid = True
            except Exception:
                if verbose:
                    print("  Output: INVALID (JSON parse error)")
                    print("  Correct: 0, Partially correct: 0, Incorrect: 0")
                else:
                    print("INVALID", end=" ")
                invalid_responses += 1
                # Track failed case for debug
                if debug:
                    trial_failed_cases.append({
                        'trial': trial,
                        'batch': entry['batch_number'],
                        'ground_truth': ground_truth,
                        'batch_data': entry['speakers'],
                        'output': output,
                        'error': 'INVALID (JSON parse error)',
                        'response_time': entry['response_time'],
                        'test_params': {
                            'llm_detects_conv_structure': llm_detects_conv_structure,
                            'use_dummy_transcripts': use_dummy_transcripts,
                            'keep_history': keep_history,
                            'num_2person_convos': num_2person_convos,
                            'num_3person_convos': num_3person_convos,
                            'num_4person_convos': num_4person_convos
                        }
                    })
                continue
            # For each conversation in the output
            batch_correct = 0
            batch_partially_correct = 0
            batch_incorrect = 0
            incorrect_conversations = []
            partially_correct_conversations = []
            for conv in output_json.values():
                speakers = set(conv.get('speakers', []))
                found = False
                for gt in ground_truth:
                    if speakers == gt:
                        batch_correct += 1
                        found = True
                        break
                if not found:
                    # Check if conversation has only 1 speaker - always classify as incorrect
                    if len(speakers) == 1:
                        batch_incorrect += 1
                        incorrect_conversations.append(speakers)
                    else:
                        # Check for partial match (at least 2 speakers in common, but not a full match)
                        # Only consider as partially correct if conversation has 2+ speakers
                        partial = False
                        for gt in ground_truth:
                            if len(speakers & gt) >= 2 and speakers != gt:
                                batch_partially_correct += 1
                                partially_correct_conversations.append(speakers)
                                partial = True
                                break
                        if not partial:
                            batch_incorrect += 1
                            incorrect_conversations.append(speakers)
            if verbose:
                print(f"  Valid: {valid}")
                print(f"  Correct: {batch_correct}, Partially correct: {batch_partially_correct}, Incorrect: {batch_incorrect}")
            else:
                print(f"C:{batch_correct} P:{batch_partially_correct} I:{batch_incorrect}")

            # Track failed case for debug if there are incorrect or partially correct classifications
            if debug and (batch_incorrect > 0 or batch_partially_correct > 0):
                error_type = []
                if batch_incorrect > 0:
                    error_type.append(f'{batch_incorrect} incorrect')
                if batch_partially_correct > 0:
                    error_type.append(f'{batch_partially_correct} partially correct')

                trial_failed_cases.append({
                    'trial': trial,
                    'batch': entry['batch_number'],
                    'ground_truth': ground_truth,
                    'batch_data': entry['speakers'],
                    'output': output,
                    'error': f'FAILED: {", ".join(error_type)} conversation(s)',
                    'incorrect_conversations': incorrect_conversations,
                    'partially_correct_conversations': partially_correct_conversations,
                    'response_time': entry['response_time'],
                    'test_params': {
                        'llm_detects_conv_structure': llm_detects_conv_structure,
                        'use_dummy_transcripts': use_dummy_transcripts,
                        'keep_history': keep_history,
                        'num_2person_convos': num_2person_convos,
                        'num_3person_convos': num_3person_convos,
                        'num_4person_convos': num_4person_convos
                    }
                })

            # Accumulate per-batch stats into per-trial stats
            total_correct += batch_correct
            total_partially_correct += batch_partially_correct
            total_incorrect += batch_incorrect
            valid_responses += 1
            total_classifications += batch_correct + batch_partially_correct + batch_incorrect

        # Print per-trial summary
        print(f"\n=== TRIAL {trial} SUMMARY ===")
        print(f"Total correct classifications:   {total_correct}")
        print(f"Total partially correct:         {total_partially_correct}")
        print(f"Total incorrect classifications: {total_incorrect}")
        print(f"Valid responses:                 {valid_responses}")
        print(f"Invalid responses:               {invalid_responses}")
        denom = total_correct + total_partially_correct + total_incorrect + invalid_responses
        if denom > 0:
            accuracy = total_correct / denom
            print(f"Average accuracy:                {accuracy:.2%}")
        else:
            print("No valid classifications to evaluate.")
        if total_batches > 0:
            avg_time = total_response_time / total_batches
            print(f"Average response time:           {avg_time:.2f} seconds")
        print("==========================================\n")

        # Store per-trial stats for file output
        all_trial_stats.append({
            'trial': trial,
            'total_correct': total_correct,
            'total_partially_correct': total_partially_correct,
            'total_incorrect': total_incorrect,
            'valid_responses': valid_responses,
            'invalid_responses': invalid_responses,
            'accuracy': accuracy if total_classifications > 0 else None,
            'avg_time': avg_time if total_batches > 0 else None,
        })

        # Add trial failed cases to main failed cases list
        failed_cases.extend(trial_failed_cases)

        # Write trial results to CSV
        write_trial_results_to_csv(csv_writer, trial, classifier.conversation_history, GROUND_TRUTH_CONVOS, verbose)
        
        # Accumulate per-trial stats into cumulative stats
        cumulative_stats['total_correct'] += total_correct
        cumulative_stats['total_partially_correct'] += total_partially_correct
        cumulative_stats['total_incorrect'] += total_incorrect
        cumulative_stats['valid_responses'] += valid_responses
        cumulative_stats['invalid_responses'] += invalid_responses
        cumulative_stats['total_classifications'] += total_classifications
        cumulative_stats['total_response_time'] += total_response_time
        cumulative_stats['total_batches'] += total_batches

        # Print cumulative summary so far after each trial
        print("\n=== CUMULATIVE SUMMARY SO FAR ===")
        print(f"Total correct classifications:   {cumulative_stats['total_correct']}")
        print(f"Total partially correct:         {cumulative_stats['total_partially_correct']}")
        print(f"Total incorrect classifications: {cumulative_stats['total_incorrect']}")
        print(f"Valid responses:                 {cumulative_stats['valid_responses']}")
        print(f"Invalid responses:               {cumulative_stats['invalid_responses']}")
        denom = cumulative_stats['total_correct'] + cumulative_stats['total_partially_correct'] + cumulative_stats['total_incorrect'] + cumulative_stats['invalid_responses']
        if denom > 0:
            accuracy = cumulative_stats['total_correct'] / denom
            print(f"Average accuracy:                {accuracy:.2%}")
        else:
            print("No valid classifications to evaluate.")
        if cumulative_stats['total_batches'] > 0:
            avg_time = cumulative_stats['total_response_time'] / cumulative_stats['total_batches']
            print(f"Average response time:           {avg_time:.2f} seconds")
        print("==========================================\n")

    # Print cumulative summary
    print("\n=== CUMULATIVE SUMMARY OVER ALL TRIALS ===")
    print(f"Total correct classifications:   {cumulative_stats['total_correct']}")
    print(f"Total partially correct:         {cumulative_stats['total_partially_correct']}")
    print(f"Total incorrect classifications: {cumulative_stats['total_incorrect']}")
    denom = cumulative_stats['total_correct'] + cumulative_stats['total_partially_correct'] + cumulative_stats['total_incorrect'] + cumulative_stats['invalid_responses']
    if denom > 0:
        accuracy = cumulative_stats['total_correct'] / denom
        print(f"Average accuracy:                {accuracy:.2%}")
    else:
        print("No valid classifications to evaluate.")
    if cumulative_stats['total_batches'] > 0:
        avg_time = cumulative_stats['total_response_time'] / cumulative_stats['total_batches']
        print(f"Average response time:           {avg_time:.2f} seconds")
    print("==========================================\n")

    # === Write results to file at the end (append or update per config) ===
    # Sanitize model name for filename
    model_name_safe = re.sub(r'[^\w\-]+', '_', model_name)
    config_header = (
        f"=== CONFIG: 2p={num_2person_convos}, 3p={num_3person_convos}, 4p={num_4person_convos} "
        f"KEEP_HISTORY={keep_history} LLM_DETECTS_CONV_STRUCTURE={llm_detects_conv_structure} USE_DUMMY_TRANSCRIPTS={use_dummy_transcripts} DEBUG={debug} ==="
    )

    # === Debug output for failed test cases ===
    debug_filename = None
    if debug and failed_cases:
        # Create debug filename
        if use_dummy_transcripts:
            debug_filename = f"debug_dummy_{model_name_safe}.txt"
        else:
            debug_filename = f"debug_{model_name_safe}.txt"

        # Prepare debug section text
        debug_section = []
        debug_section.append(f"\n{config_header}\n")
        debug_section.append(f"LLM Model: {model_name}\n")
        debug_section.append(f"Number of trials (this run): {trials}\n")
        debug_section.append(f"Total failed cases: {len(failed_cases)}\n")
        debug_section.append(f"\n=== FAILED TEST CASES ===\n")

        for i, case in enumerate(failed_cases, 1):
            debug_section.append(f"\n--- FAILED CASE {i} ---\n")
            debug_section.append(f"Trial: {case['trial']}\n")
            debug_section.append(f"Batch: {case['batch']}\n")
            debug_section.append(f"Response Time: {case['response_time']:.2f}s\n")
            debug_section.append(f"Error: {case['error']}\n")
            debug_section.append("\nTEST PARAMETERS:\n")
            params = case['test_params']
            debug_section.append(f"  LLM Detects Conv Structure: {params['llm_detects_conv_structure']}\n")
            debug_section.append(f"  Use Dummy Transcripts: {params['use_dummy_transcripts']}\n")
            debug_section.append(f"  Keep History: {params['keep_history']}\n")
            debug_section.append(f"  2-person conversations: {params['num_2person_convos']}\n")
            debug_section.append(f"  3-person conversations: {params['num_3person_convos']}\n")
            debug_section.append(f"  4-person conversations: {params['num_4person_convos']}\n")
            debug_section.append("\nGROUND TRUTH CONVERSATION GROUPINGS:\n")
            for j, group in enumerate(case['ground_truth'], 1):
                debug_section.append(f"  Conversation {j}: {{ {', '.join(sorted(group))} }}\n")
            if 'incorrect_conversations' in case and case['incorrect_conversations']:
                debug_section.append("\nINCORRECT CONVERSATIONS DETECTED:\n")
                for j, conv in enumerate(case['incorrect_conversations'], 1):
                    debug_section.append(f"  {j}: {{ {', '.join(sorted(conv))} }}\n")
            if 'partially_correct_conversations' in case and case['partially_correct_conversations']:
                debug_section.append("\nPARTIALLY CORRECT CONVERSATIONS DETECTED:\n")
                for j, conv in enumerate(case['partially_correct_conversations'], 1):
                    debug_section.append(f"  {j}: {{ {', '.join(sorted(conv))} }}\n")
            debug_section.append("\nBATCH TRANSCRIPTION:\n")
            for speaker_id, lines in case['batch_data'].items():
                transcript = " ".join(lines)
                debug_section.append(f"  {speaker_id}: {transcript}\n")
            debug_section.append("\nLLM OUTPUT:\n")
            debug_section.append(f"{case['output']}\n")
            debug_section.append("-" * 60 + "\n")

        # Write debug file
        with open(debug_filename, 'a', encoding='utf-8') as f:
            f.write(''.join(debug_section))
        print(f"\nDebug cases saved to {debug_filename}\n")

    # Create separate filename for dummy tests
    if use_dummy_transcripts:
        results_filename = f"results_dummy_{model_name_safe}.txt"
    else:
        results_filename = f"results_{model_name_safe}.txt"
    # Prepare new section text
    new_section = []
    new_section.append(f"\n{config_header}\n")
    new_section.append(f"LLM Model: {model_name}\n")
    new_section.append(f"Number of trials (this run): {trials}\n")
    new_section.append(f"KEEP_HISTORY: {keep_history}\n")
    new_section.append(f"LLM_DETECTS_CONV_STRUCTURE: {llm_detects_conv_structure}\n")
    new_section.append(f"USE_DUMMY_TRANSCRIPTS: {use_dummy_transcripts}\n")
    new_section.append(f"DEBUG: {debug}\n")
    new_section.append(f"\n=== PARAMETERS ===\n")
    new_section.append(f"model_name = {model_name}\n")
    new_section.append(f"trials = {trials}\n")
    new_section.append(f"num_2person_convos = {num_2person_convos}\n")
    new_section.append(f"num_3person_convos = {num_3person_convos}\n")
    new_section.append(f"num_4person_convos = {num_4person_convos}\n")
    new_section.append(f"llm_detects_conv_structure = {llm_detects_conv_structure}\n")
    new_section.append(f"use_dummy_transcripts = {use_dummy_transcripts}\n")
    new_section.append(f"keep_history = {keep_history}\n")
    new_section.append(f"debug = {debug}\n")
    new_section.append(f"\n=== CUMULATIVE SUMMARY OVER ALL TRIALS ===\n")
    new_section.append(f"Total correct classifications:   {cumulative_stats['total_correct']}\n")
    new_section.append(f"Total partially correct:         {cumulative_stats['total_partially_correct']}\n")
    new_section.append(f"Total incorrect classifications: {cumulative_stats['total_incorrect']}\n")
    new_section.append(f"Valid responses:                 {cumulative_stats['valid_responses']}\n")
    new_section.append(f"Invalid responses:               {cumulative_stats['invalid_responses']}\n")
    denom = cumulative_stats['total_correct'] + cumulative_stats['total_partially_correct'] + cumulative_stats['total_incorrect'] + cumulative_stats['invalid_responses']
    if denom > 0:
        accuracy = cumulative_stats['total_correct'] / denom
        new_section.append(f"Average accuracy:                {accuracy:.2%}\n")
    else:
        new_section.append("No valid classifications to evaluate.\n")
    if cumulative_stats['total_batches'] > 0:
        avg_time = cumulative_stats['total_response_time'] / cumulative_stats['total_batches']
        new_section.append(f"Average response time:           {avg_time:.2f} seconds\n")
    new_section.append("==========================================\n")

    # Append to file
    with open(results_filename, 'a', encoding='utf-8') as f:
        f.write(''.join(new_section))
    print(f"\nResults appended to {results_filename}\n")
    
    # Write final summary to CSV
    csv_writer.writerow([])  # Empty row for spacing
    csv_writer.writerow(["FINAL SUMMARY"])
    csv_writer.writerow(["Metric", "Value"])
    csv_writer.writerow(["Total Correct", cumulative_stats['total_correct']])
    csv_writer.writerow(["Total Partially Correct", cumulative_stats['total_partially_correct']])
    csv_writer.writerow(["Total Incorrect", cumulative_stats['total_incorrect']])
    csv_writer.writerow(["Valid Responses", cumulative_stats['valid_responses']])
    csv_writer.writerow(["Invalid Responses", cumulative_stats['invalid_responses']])
    denom = cumulative_stats['total_correct'] + cumulative_stats['total_partially_correct'] + cumulative_stats['total_incorrect'] + cumulative_stats['invalid_responses']
    if denom > 0:
        accuracy = cumulative_stats['total_correct'] / denom
        csv_writer.writerow(["Overall Accuracy", f"{accuracy:.2%}"])
    if cumulative_stats['total_batches'] > 0:
        avg_time = cumulative_stats['total_response_time'] / cumulative_stats['total_batches']
        csv_writer.writerow(["Average Response Time (seconds)", f"{avg_time:.2f}"])
    
    print(f"\nCSV results saved to {csv_filename}\n")
    
    # Close CSV file
    csvfile.close()

def create_csv_filename(model_name: str, trials: int, num_2person_convos: int, num_3person_convos: int, 
                       num_4person_convos: int, llm_detects_conv_structure: bool, use_dummy_transcripts: bool, 
                       keep_history: bool, debug: bool) -> str:
    """Create a condensed filename for CSV output with test parameters"""
    # Sanitize model name for filename
    model_name_safe = re.sub(r'[^\w\-]+', '_', model_name)
    
    # Create condensed parameter string - only include specified parameters when True
    params = []
    
    if llm_detects_conv_structure:
        params.append("detect")
    if use_dummy_transcripts:
        params.append("dummy")
    if keep_history:
        params.append("history")
    
    param_str = "_".join(params) if params else "default"
    
    return f"results_{model_name_safe}_{param_str}.csv"

def write_trial_results_to_csv(csv_writer, trial_number: int, conversation_history: List[Dict], 
                              ground_truth: List[set], verbose: bool = False):
    """Write trial results to CSV with batch-by-batch classification accuracy"""
    
    # Write trial header
    csv_writer.writerow([])  # Empty row for spacing
    csv_writer.writerow([f"TRIAL {trial_number}"])
    csv_writer.writerow(["Batch", "Correct", "Partially_Correct", "Incorrect", "Invalid", "Response_Time"])
    
    # Process each batch in the conversation history
    for entry in conversation_history:
        batch_number = entry['batch_number']
        output = entry['classification']
        response_time = entry['response_time']
        
        # Initialize counters
        correct = 0
        partially_correct = 0
        incorrect = 0
        invalid = 0
        
        # Find the first {...} block (the JSON)
        match = re.search(r'\{[\s\S]*\}', output)
        if not match:
            invalid = 1
        else:
            try:
                output_json = json.loads(match.group(0))
                # Check for duplicate speaker assignments
                all_speakers = []
                for conv in output_json.values():
                    all_speakers.extend(conv.get('speakers', []))
                if len(all_speakers) != len(set(all_speakers)):
                    invalid = 1
                else:
                    # For each conversation in the output
                    for conv in output_json.values():
                        speakers = set(conv.get('speakers', []))
                        found = False
                        for gt in ground_truth:
                            if speakers == gt:
                                correct += 1
                                found = True
                                break
                        if not found:
                            # Check if conversation has only 1 speaker - always classify as incorrect
                            if len(speakers) == 1:
                                incorrect += 1
                            else:
                                # Check for partial match (at least 2 speakers in common, but not a full match)
                                # Only consider as partially correct if conversation has 2+ speakers
                                partial = False
                                for gt in ground_truth:
                                    if len(speakers & gt) >= 2 and speakers != gt:
                                        partially_correct += 1
                                        partial = True
                                        break
                                if not partial:
                                    incorrect += 1
            except Exception:
                invalid = 1
        
        # Write batch results
        csv_writer.writerow([batch_number, correct, partially_correct, incorrect, invalid, f"{response_time:.2f}"])

def main():
    """Main function with default configuration - calls run_classification_test with current macro values"""

    # Default configuration (previously defined as macros)
    LLM_MODEL_NAME = 'Qwen/Qwen3-0.6B'
    TRIALS = 3
    NUM_2PERSON_CONVOS = 1
    NUM_3PERSON_CONVOS = 1
    NUM_4PERSON_CONVOS = 0
    LLM_DETECTS_CONV_STRUCTURE = False
    USE_DUMMY_TRANSCRIPTS = False
    KEEP_HISTORY = False
    VERBOSE = True
    DEBUG = False

    # Run the test with default configuration
    run_classification_test(
        model_name=LLM_MODEL_NAME,
        trials=TRIALS,
        num_2person_convos=NUM_2PERSON_CONVOS,
        num_3person_convos=NUM_3PERSON_CONVOS,
        num_4person_convos=NUM_4PERSON_CONVOS,
        llm_detects_conv_structure=LLM_DETECTS_CONV_STRUCTURE,
        use_dummy_transcripts=USE_DUMMY_TRANSCRIPTS,
        keep_history=KEEP_HISTORY,
        verbose=VERBOSE,
        debug=DEBUG
    )

if __name__ == "__main__":
    main() 
