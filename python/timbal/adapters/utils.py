from datetime import datetime
import os

from config import logger
from call_transcript_analyzer import analyze_transcript


async def generate_full_call_transcript(call_session):
    """Generate a complete call transcript at the end."""
    try:
        summary = call_session.get_conversation_summary()
        
        print("\n" + "=" * 80)
        print("📞 COMPLETE CALL TRANSCRIPT")
        print("=" * 80)
        print(f"📱 Call SID: {summary['call_sid']}")
        print(f"⏰ Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"🔄 Total exchanges: {summary['total_exchanges']}")
        print("-" * 80)
        
        if summary['conversation_history']:
            print("💬 CONVERSATION:")
            for entry in summary['conversation_history']:
                # Convert ISO string back to datetime object if needed
                timestamp_str = entry['timestamp']
                if isinstance(timestamp_str, str):
                    dt_object = datetime.fromisoformat(timestamp_str)
                else:
                    dt_object = timestamp_str
                
                timestamp = dt_object.strftime('%H:%M:%S')
                speaker_icon = "👤" if entry['speaker'] == "user" else "🤖"
                speaker_name = "USER" if entry['speaker'] == "user" else "AGENT"
                print(f"[{timestamp}] {speaker_icon} {speaker_name}: {entry['text']}")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"💥 Error generating transcript: {e}")


async def analyze_call_transcript(call_session):
    """Analyze the call transcript for customer satisfaction survey results."""
    try:
        # Prepare agent state in the format expected by analyze_transcript
        agent_state = {
            "conversation_history": [
                {
                    "speaker": entry["speaker"],
                    "text": entry["text"],
                    "timestamp": entry["timestamp"]
                }
                for entry in call_session.conversation_history
            ]
        }
        
        logger.info("🔍 Starting satisfaction survey analysis...")
        
        # Call our imported function
        analysis_result = await analyze_transcript(agent_state)
        print(analysis_result)
        
        # Display results
        print("\n" + "📊 " + "=" * 78)
        print("ANÁLISIS DE ENCUESTA DE SATISFACCIÓN - VICIO")
        print("=" * 80)
        
        if analysis_result:
            # Survey status
            if analysis_result.get("survey_completed"):
                print("✅ ENCUESTA COMPLETADA EXITOSAMENTE")
            else:
                print("❌ ENCUESTA NO COMPLETADA")

            # Rating
            rating = analysis_result.get("satisfaction_rating")
            if rating is not None:
                print(f"   ⭐️ Puntuación: {rating}/5")
            
            # Reason
            reason = analysis_result.get("reason_for_rating")
            if reason:
                print(f"   🗣️ Motivo: {reason}")

            # Customer response
            customer_response = analysis_result.get("customer_response", "incomplete")
            response_icons = {
                "completed": "✅",
                "declined": "❌", 
                "incomplete": "❓"
            }
            icon = response_icons.get(customer_response, "❓")
            print(f"   {icon} Respuesta del cliente: {customer_response}")
            
            # Call outcome
            call_outcome = analysis_result.get("call_outcome", "unclear")
            outcome_icons = {
                "successful_survey": "🎯",
                "declined": "❌",
                "no_rating_provided": "🔇",
                "unclear": "❓"
            }
            outcome_icon = outcome_icons.get(call_outcome, "❓")
            print(f"   {outcome_icon} Resultado de la llamada: {call_outcome}")
                
        else:
            print("❌ No se pudo analizar la conversación")
        
        print("=" * 80)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"💥 Error analyzing appointment scheduling: {e}")
        print(f"\n❌ ERROR EN ANÁLISIS DE ENCUESTA: {e}")
        return None


async def save_agent_memory_to_file(call_session, call_sid: str):
    """Save the agent's memory to a TXT file when call finishes"""
    try:
        # Define the directory and ensure it exists
        log_dir = "call_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create filename with path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        call_sid_short = call_sid[:8] if call_sid else "unknown"
        filename = f"call_memory_{call_sid_short}_{timestamp}.txt"
        filepath = os.path.join(log_dir, filename)

        # Save to TXT file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("CALL MEMORY DUMP\n")
            f.write("=" * 40 + "\n\n")
            
            # Write metadata
            f.write("CALL METADATA:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Call SID: {call_sid}\n")
            f.write(f"Start Time: {call_session.start_time.isoformat()}\n")
            f.write(f"End Time: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {(datetime.now() - call_session.start_time).total_seconds():.1f} seconds\n")
            f.write(f"Total Exchanges: {call_session.conversation_count}\n\n")
            
            # Write conversation
            f.write("CONVERSATION HISTORY:\n")
            f.write("-" * 20 + "\n")
            for entry in call_session.conversation_history:
                speaker = "USER" if entry['speaker'] == 'user' else "AGENT"
                timestamp_str = entry['timestamp']
                # Ensure timestamp is a string for writing
                if isinstance(timestamp_str, datetime):
                    ts = timestamp_str.isoformat()
                else:
                    ts = str(timestamp_str)
                f.write(f"[{ts[:19]}] {speaker}: {entry['text']}\n")

        logger.info(f"✅ Memory saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving memory: {e}")
        return None


async def display_analysis(agent_state: dict):
    """Fetches and displays the call analysis for the satisfaction survey."""
    print("\n\n" + "="*20)
    print("ANALIZANDO TRANSCRIPCIÓN DE ENCUESTA...")
    print("="*20)

    analysis = await analyze_transcript(agent_state)

    if not analysis:
        print("No se pudo obtener el análisis.")
        return

    print("\n--- ANÁLISIS DE LA ENCUESTA DE SATISFACCIÓN ---")
    if "error" in analysis:
        print(f"  Error en el análisis: {analysis['error']}")
    else:
        survey_completed = analysis.get('survey_completed', False)
        rating = analysis.get('satisfaction_rating')
        reason = analysis.get('reason_for_rating') or 'No proporcionado'
        customer_response = analysis.get('customer_response') or 'incompleta'
        outcome = analysis.get('call_outcome', 'Desconocido')

        print(f"  - Encuesta completada: {'Sí' if survey_completed else 'No'}")
        if rating is not None:
            print(f"  - Puntuación: {rating}/5")
        else:
            print(f"  - Puntuación: No proporcionada")
        print(f"  - Motivo: {reason}")
        print(f"  - Respuesta del cliente: {customer_response}")
        print(f"  - Resultado de la llamada: {outcome}")
    print("-------------------------------------------\n")


def display_call_start():
    """Prints a message indicating the call has started."""
    print("\n" + "=" * 80)
    print("📞 CALL STARTED")
    print("=" * 80) 