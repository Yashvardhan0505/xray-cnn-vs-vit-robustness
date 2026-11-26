"""
main.py
B.Tech Project: Robustness Analysis of CNNs vs ViTs in Medical Imaging
Description: Master orchestrator script to run the entire pipeline sequentially.
"""

import subprocess
import sys
import time

def run_script(script_name):
    """Executes a python script and handles potential errors."""
    print(f"\n{'='*70}")
    print(f"🚀 STARTING STAGE: {script_name}")
    print(f"{'='*70}")
    
    try:
        # sys.executable ensures it uses the exact same Python environment (e.g., your gpu_env)
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ STAGE COMPLETED: {script_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_name} crashed with exit code {e.returncode}.")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find '{script_name}'. Ensure it is in the same folder as main.py.")
        return False

def main():
    print("\n" + "🌟"*35)
    print("   B.TECH PROJECT PIPELINE ORCHESTRATOR")
    print("   Robustness Analysis of CNNs vs ViTs")
    print("🌟"*35 + "\n")
    
    # The exact order your scripts must run in
    pipeline_scripts = [
        "1_data_preparation.py",
        "2_train_and_robustness.py",
        "3_custom_cnn.py",
        "4_xai_gradcam.py"
    ]
    
    start_time = time.time()
    
    for script in pipeline_scripts:
        success = run_script(script)
        if not success:
            print("\n🛑 PIPELINE HALTED: Please fix the error above before continuing.")
            sys.exit(1)
            
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print(f"🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f} MINUTES! 🎉")
    print("Check your directory for the generated .pth models and .pdf Paper Plots.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
