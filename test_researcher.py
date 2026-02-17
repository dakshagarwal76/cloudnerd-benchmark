"""
Test script to evaluate deep_researcher on all questions from so_qna.csv
Enhanced with checkpointing, logging, and comprehensive summary
"""
import sys
import io

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import asyncio
import json
import os
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from fix_hallucinations import AnswerValidator

# FIXED: Import with correct package path
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import Configuration, SearchAPI
from langchain_core.runnables import RunnableConfig

class ResearcherTester:
    def __init__(self,
                 questions_file: str = "mapped_rag_test_questions.csv",
                 output_file: str = "test_results.csv",
                 checkpoint_file: str = "test_checkpoint.json",
                 log_file: str = "test_log.txt"):
        """Initialize the tester with input, output, checkpoint, and log files."""
        self.questions_file = questions_file
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.log_file = log_file
        self.results = []
        self.graph = deep_researcher
        
        # Setup logging
        self.setup_logging()

        self.validator = AnswerValidator('cloud.jsonl')
        
        # Load checkpoint if exists
        self.start_index = self.load_checkpoint()
    
    def setup_logging(self):
        """Setup logging to both file and console."""
        self.logger = logging.getLogger('ResearcherTester')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.propagate = False
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            self.log_file, 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler with UTF-8 - ADD THIS
        console_handler = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        )
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def load_checkpoint(self) -> int:
        """Load checkpoint to resume from last processed question."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    last_index = checkpoint_data.get('last_completed_index', -1)
                    self.logger.info(f" Checkpoint found. Resuming from question {last_index + 2}")
                    return last_index + 1
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
                return 0
        else:
            self.logger.info("No checkpoint found. Starting from beginning.")
            return 0
    
    def save_checkpoint(self, index: int, question_id: int):
        """Save checkpoint after processing each question."""
        checkpoint_data = {
            'last_completed_index': index,
            'last_completed_question_id': question_id,
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.results)
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_questions(self) -> pd.DataFrame:
        """Load questions from CSV file."""
        df = pd.read_csv(self.questions_file)
        self.logger.info(f"Loaded {len(df)} questions from {self.questions_file}")
        return df
    
    async def test_single_question(self, question_id: int, question: str, posted_date: str, index: int, total: int) -> Dict:
        """
        Test a single question through the deep researcher.
        
        Args:
            question_id: The ID of the question
            question: The question text
            index: Current question index (for progress tracking)
            total: Total number of questions
            
        Returns:
            Dictionary with test results including hallucination detection
        """
        self.logger.info("")
        self.logger.info(f"[{index+1}/{total}] Testing Question ID: {question_id}")
        
        try:
            # Create configuration - JSONL only, no external data access
            config = RunnableConfig(
                configurable={
                    "search_api": "jsonl",
                    "max_search_queries": 3,
                    "max_content_length": 10000,
                    "allow_clarification": False,
                }
            )
            
            # Run the question through the system
            start_time = datetime.now()
            
            # Invoke the graph with the question
            # === MODIFIED: Include posted_date in the input ===
            enhanced_question = f"[Question posted on: {posted_date}]\n\n{question}"

            # Invoke the graph with the enhanced question
            result = await self.graph.ainvoke(
                {"messages": [("user", enhanced_question)]},
                config=config
            )
            # === END MODIFICATION ===

            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract the answer from the result
            generated_answer = ""
            
            # Try to get from messages first
            if "messages" in result and len(result["messages"]) > 0:
                # Try to get the last AI message with substantial content
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'content'):
                        if msg.content and len(msg.content) > 50:
                            generated_answer = msg.content
                            break
                
                # Fallback: get any message with content
                if not generated_answer:
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, 'content') and msg.content:
                            generated_answer = str(msg.content)
                            break
            
            # Check for final_report if available
            if not generated_answer and "final_report" in result:
                generated_answer = result["final_report"]
            
            # Check for report field
            if not generated_answer and "report" in result:
                generated_answer = result["report"]
            
            # === NEW: VALIDATE AND CLEAN ANSWER ===
            validation_result = self.validator.process_answer(generated_answer)
            generated_answer = validation_result['cleaned_text']
            
            # Log validation issues
            # In test_single_question method, after validation:

            if validation_result['has_hallucinations']:
                self.logger.warning(f"    HALLUCINATION DETECTED!")
                self.logger.warning(f"     Fake Post IDs: {validation_result['fake_citations']}")
                
                if validation_result['valid_citations']:
                    self.logger.warning(f"     Valid Post IDs: {validation_result['valid_citations']}")
                else:
                    self.logger.warning(f"     CRITICAL: No valid citations found - ALL citations were fake!")
            
            # === END VALIDATION ===
            
            has_answer = len(generated_answer) > 50
            
            # Log success with validation stats
            self.logger.info(f" SUCCESS - Completed in {duration:.2f} seconds")
            self.logger.info(f"  Answer generated: {'YES' if has_answer else 'NO'}")
            
            return {
                "question_id": question_id,
                "question": question,
                "generated_answer": generated_answer,
                "posted_date": posted_date,

            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Log full traceback to file only
            import traceback
            full_traceback = traceback.format_exc()
            
            # Get file handler and write traceback
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.stream.write(f"\nFull traceback:\n{full_traceback}\n")
                    handler.flush()
                    break
            
            return {
                "question_id": question_id,
                "question": question,
                "generated_answer": ""
            }



    async def test_all_questions(self, limit: int = None):
        """
        Test all questions from the CSV file with checkpointing.
        
        Args:
            limit: Optional limit on number of questions to test
        """
        # Load questions
        df = self.load_questions()
        
        # Apply limit if specified
        if limit:
            df = df.head(limit)
            self.logger.info(f"Testing limited to first {limit} questions")
        
        total = len(df)
        
        # Load existing results if resuming
        if os.path.exists(self.output_file) and self.start_index > 0:
            try:
                existing_df = pd.read_csv(self.output_file)
                self.results = existing_df.to_dict('records')
                self.logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")
        
        # Process each question starting from checkpoint
        for idx, row in df.iterrows():
            # Skip already processed questions
            if idx < self.start_index:
                continue
            
            result = await self.test_single_question(
                question_id=row['question_id'],
                question=row['question'],
                posted_date=row['posted_date'],
                index=idx,
                total=total
            )
            
            # Add original answers for later evaluation
            result['answer_raw'] = row.get('answer_raw', '')
            result['answer_summarized'] = row.get('answer_summarized', '')
            result['posted_date'] = row['posted_date']


            
            self.results.append(result)
            
            # Save results immediately after each question
            self.save_results()
            
            # Save checkpoint
            self.save_checkpoint(idx, row['question_id'])
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(1)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"TESTING COMPLETE! Processed {len(self.results)} questions")
        self.logger.info(f"Results saved to {self.output_file}")
        self.logger.info("="*70)
    
    def save_results(self):
        """Save results to CSV file."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_file, index=False)
    
    def print_detailed_summary(self):
        """Print comprehensive summary statistics including hallucination analysis."""
        if not self.results:
            self.logger.info("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        total = len(df)
        successful = len(df[df['status'] == 'success'])
        failed = len(df[df['status'] == 'error'])
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("DETAILED SUMMARY REPORT")
        self.logger.info("="*70)
        
        # Overall Statistics
        self.logger.info("\n📊 OVERALL STATISTICS")
        self.logger.info(f"  Total questions tested: {total}")
        self.logger.info(f"  Successful runs: {successful} ({successful/total*100:.1f}%)")
        self.logger.info(f"  Failed runs: {failed} ({failed/total*100:.1f}%)")
        
        # Success Analysis
        if successful > 0:
            success_df = df[df['status'] == 'success']
            
            # Answer generation stats
            with_answers = success_df['has_answer'].sum()
            without_answers = successful - with_answers
            
            # === END HALLUCINATION ANALYSIS ===
            
            # Timing statistics
            self.logger.info("\n⏱️ TIMING STATISTICS")
            self.logger.info(f"  Average duration: {success_df['duration_seconds'].mean():.2f} seconds")
            self.logger.info(f"  Total time: {success_df['duration_seconds'].sum():.2f} seconds ({success_df['duration_seconds'].sum()/60:.2f} minutes)")
            
        # Error Analysis (rest stays the same)
        if failed > 0:
            error_df = df[df['status'] == 'error']
            
            self.logger.info("\n ERROR ANALYSIS")
            self.logger.info(f"  Total errors: {failed}")
            
            # Group errors by type
            error_types = error_df['error'].value_counts()
            self.logger.info("\n  Error types:")
            for error_type, count in error_types.items():
                error_preview = error_type[:80] + "..." if len(error_type) > 80 else error_type
                self.logger.info(f"    • {error_preview}: {count} occurrence(s)")
            
            # List failed question IDs
            self.logger.info("\n  Failed question IDs:")
            failed_ids = error_df['question_id'].tolist()
            self.logger.info(f"    {failed_ids}")
        
        # Output file information
        self.logger.info("\n OUTPUT FILES")
        self.logger.info(f"  Results CSV: {self.output_file}")
        self.logger.info(f"  Log file: {self.log_file}")
        self.logger.info(f"  Checkpoint: {self.checkpoint_file}")
        
        self.logger.info("\n" + "="*70)
        self.logger.info("SESSION COMPLETED: " + datetime.now().isoformat())
        self.logger.info("="*70)

async def main():
    """Main function to run the tests."""
    print("="*70)
    print("DEEP RESEARCHER TESTING ")
    print("="*70)
    
    # Create tester instance
    tester = ResearcherTester(
        questions_file="mapped_rag_test_questions.csv",
        output_file="test_results.csv",
        checkpoint_file="test_checkpoint.json",
        log_file="test_log.txt"
    )
    
    # Run tests
    # OPTION 1: Test all questions
    await tester.test_all_questions(limit=100)
    
    # Print detailed summary
    tester.print_detailed_summary()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n Testing interrupted by user. Progress has been saved.")
        print("Run the script again to resume from the last checkpoint.")