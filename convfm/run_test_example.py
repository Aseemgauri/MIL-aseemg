#!/usr/bin/env python3
"""
Example script demonstrating how to run multi-speaker conversation classification tests
with different configurations using the refactored classifier.
"""

from multi_speaker_conversation_classifier import run_classification_test
import itertools

def run_all_combinations(model_name, trials, num_2person_convos, num_3person_convos, num_4person_convos, verbose=True):
    """
    Run all possible combinations of test configurations with the given conversation setup.
    
    Args:
        model_name: The LLM model to use
        trials: Number of trials for each test
        num_2person_convos: Number of 2-person conversations
        num_3person_convos: Number of 3-person conversations  
        num_4person_convos: Number of 4-person conversations
    """
    
    print(f"🚀 Running ALL combinations for {model_name}")
    print(f"📊 Configuration: {num_2person_convos} 2p, {num_3person_convos} 3p, {num_4person_convos} 4p conversations")
    print(f"🔄 Trials per test: {trials}")
    print("=" * 80)
    
    # Define all possible boolean combinations
    llm_detects_options = [False, True]
    use_dummy_options = [False, True] 
    keep_history_options = [False, True]
    
    # Generate all combinations
    combinations = list(itertools.product(llm_detects_options, use_dummy_options, keep_history_options))
    
    print(f"📋 Total combinations to test: {len(combinations)}")
    print()
    
    for i, (llm_detects, use_dummy, keep_history) in enumerate(combinations, 1):
        print(f"\n🔬 Test {i}/{len(combinations)}")
        print(f"   LLM detects structure: {llm_detects}")
        print(f"   Use dummy transcripts: {use_dummy}")
        print(f"   Keep history: {keep_history}")
        print("-" * 50)
        
        try:
            run_classification_test(
                model_name=model_name,
                trials=trials,
                num_2person_convos=num_2person_convos,
                num_3person_convos=num_3person_convos,
                num_4person_convos=num_4person_convos,
                llm_detects_conv_structure=llm_detects,
                use_dummy_transcripts=use_dummy,
                keep_history=keep_history,
                verbose=verbose,
                debug=True
            )
            print(f"✅ Test {i} completed successfully")
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
        
        print()
    
    print("🎉 All combination tests completed!")
    print("📊 Check the results files for detailed statistics")

def main():
    #model = 'Qwen/Qwen3-0.6B'
    #model = 'o4-mini'
    model = 'gpt-4o'
    #model = 'gpt-4o-mini'
    #model = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    #model = 'claude-3-5-haiku-latest'
    #model = 'gemini-2.5-flash'

    '''
    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=3,
        num_3person_convos=0,
        num_4person_convos=0,
        llm_detects_conv_structure=False,
        use_dummy_transcripts=False,
        keep_history=False,
        verbose=True,
        debug=True
    )
    '''

    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=3,
        num_3person_convos=0,
        num_4person_convos=0,
        llm_detects_conv_structure=True,
        use_dummy_transcripts=False,
        keep_history=True,
        verbose=True,
        debug=True
    )

    '''
    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=1,
        num_3person_convos=1,
        num_4person_convos=0,
        llm_detects_conv_structure=False,
        use_dummy_transcripts=False,
        keep_history=False,
        verbose=True,
        debug=True
    )
    '''

    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=1,
        num_3person_convos=1,
        num_4person_convos=0,
        llm_detects_conv_structure=True,
        use_dummy_transcripts=False,
        keep_history=True,
        verbose=True,
        debug=True
    )

    '''
    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=0,
        num_3person_convos=2,
        num_4person_convos=0,
        llm_detects_conv_structure=False,
        use_dummy_transcripts=False,
        keep_history=False,
        verbose=True,
        debug=True
    )
    '''

    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=0,
        num_3person_convos=2,
        num_4person_convos=0,
        llm_detects_conv_structure=True,
        use_dummy_transcripts=False,
        keep_history=True,
        verbose=True,
        debug=True
    )

    '''
    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=1,
        num_3person_convos=0,
        num_4person_convos=1,
        llm_detects_conv_structure=False,
        use_dummy_transcripts=False,
        keep_history=False,
        verbose=True,
        debug=True
    )
    '''

    run_classification_test(
        model_name=model,
        trials=10,
        num_2person_convos=1,
        num_3person_convos=0,
        num_4person_convos=1,
        llm_detects_conv_structure=True,
        use_dummy_transcripts=False,
        keep_history=True,
        verbose=True,
        debug=True
    )


if __name__ == "__main__":
    main() 