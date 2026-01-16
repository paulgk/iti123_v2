"""
LLM-Enhanced Coaching Feedback
Combines rule-based biomechanical analysis with natural language generation
"""

import os
from typing import Dict, List, Optional
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. LLM enhancement disabled.")

class LLMCoachingEnhancer:
    """
    Enhances rule-based feedback with LLM-generated natural language

    Uses OpenAI GPT-4o-mini for:
    - Natural, conversational feedback
    - Context-aware recommendations
    - Personalized tone based on skill level
    - Creative practice drills
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM enhancer

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # Initialize OpenAI client (v1.0+ API)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except:
            # Fallback for older openai package
            openai.api_key = self.api_key
            self.client = None

        self.model = "gpt-4o-mini"  # Fast, cheap, high quality

    def enhance_feedback(
        self,
        feedback_items: List[Dict],
        stroke_type: str,
        overall_score: int,
        player_level: Optional[str] = None
    ) -> str:
        """
        Generate enhanced coaching feedback using LLM

        Args:
            feedback_items: List of rule-based feedback items
            stroke_type: "Clear" or "Smash"
            overall_score: Technique score (0-100)
            player_level: "beginner", "intermediate", or "advanced"

        Returns:
            HTML-formatted coaching feedback
        """
        # Auto-detect level if not provided
        if player_level is None:
            player_level = self._estimate_level(overall_score)

        # Build structured prompt
        prompt = self._build_prompt(
            feedback_items, stroke_type, overall_score, player_level
        )

        # Call LLM with retry logic
        try:
            enhanced_text = self._call_llm(prompt)
            return self._format_as_html(enhanced_text, overall_score, stroke_type)

        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            # Fallback to template-based
            return self._fallback_format(feedback_items, overall_score)

    def _estimate_level(self, score: int) -> str:
        """Estimate player level from technique score"""
        if score >= 85:
            return "advanced"
        elif score >= 70:
            return "intermediate"
        else:
            return "beginner"

    def _build_prompt(
        self,
        feedback_items: List[Dict],
        stroke_type: str,
        overall_score: int,
        player_level: str
    ) -> str:
        """Build structured prompt for LLM"""

        # System context
        system_context = f"""You are analyzing a {player_level} player's {stroke_type} stroke.
Overall technique score: {overall_score}/100

Biomechanical issues identified:
"""

        # Add top 5 issues with data
        for i, item in enumerate(feedback_items[:5], 1):
            system_context += f"""
{i}. {item['category']}: {item['issue']}
   Current: {item['current_value']}
   Target: {item['target_range']}
   Severity: {item['severity']}
"""

        # Instructions
        instructions = """
Generate a comprehensive coaching report with:

1. **Opening** (1-2 sentences):
   - Acknowledge overall performance
   - Set encouraging tone

2. **Key Issues** (top 3 priorities):
   For each issue:
   - Explain clearly in simple terms
   - Why it matters (performance impact)
   - How to fix it (specific technique adjustment)

3. **Practice Plan** (this week):
   - 2-3 specific drills with sets/reps
   - Progressive difficulty
   - How to self-check improvement

4. **Closing** (1 sentence):
   - Realistic timeline for improvement
   - Motivational note

Style guidelines:
- Use second person ("you", "your")
- Be encouraging but honest
- Avoid jargon - explain biomechanics simply
- Use analogies/metaphors when helpful
- Add relevant emojis sparingly (max 3-4)
- Keep total length under 400 words

Format as clean markdown.
"""

        return system_context + instructions

    def _call_llm(self, prompt: str, max_retries: int = 2) -> str:
        """Call OpenAI API with retry logic"""

        system_prompt = """You are a professional badminton coach with 15 years of experience.

Your coaching philosophy:
- Start with what they're doing well
- Explain issues in terms of performance outcomes
- Give specific, actionable fixes
- Provide realistic timelines
- Stay encouraging and supportive

Your expertise:
- Biomechanics of overhead strokes
- Progressive skill development
- Sports psychology (motivation)
- Injury prevention
"""

        for attempt in range(max_retries + 1):
            try:
                if self.client:  # New OpenAI API (v1.0+)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800,
                        timeout=10.0
                    )
                    return response.choices[0].message.content

                else:  # Legacy OpenAI API
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    return response.choices[0].message.content

            except Exception as e:
                if attempt < max_retries:
                    print(f"LLM call failed (attempt {attempt+1}/{max_retries+1}): {e}")
                    continue
                else:
                    raise

    def _format_as_html(self, markdown_text: str, score: int, stroke_type: str) -> str:
        """Convert markdown to styled HTML"""

        try:
            import markdown
            html_content = markdown.markdown(
                markdown_text,
                extensions=['fenced_code', 'nl2br']
            )
        except ImportError:
            # Fallback: simple markdown parsing
            html_content = self._simple_markdown_to_html(markdown_text)

        # Determine score color
        if score >= 85:
            score_color = "#28a745"  # Green
            grade = "Excellent"
        elif score >= 70:
            score_color = "#ffc107"  # Yellow
            grade = "Good"
        elif score >= 60:
            score_color = "#fd7e14"  # Orange
            grade = "Fair"
        else:
            score_color = "#dc3545"  # Red
            grade = "Needs Work"

        # Create styled HTML
        styled_html = f"""
        <div style="font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                    line-height: 1.7; max-width: 900px; margin: 0 auto;">

            <!-- Score Card -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 30px; border-radius: 15px; color: white;
                        margin-bottom: 30px; text-align: center;">
                <h2 style="margin: 0 0 15px 0; font-size: 28px;">
                    🏸 {stroke_type} Technique Analysis
                </h2>
                <div style="display: flex; align-items: center; justify-content: center;
                           gap: 30px; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 56px; font-weight: bold; color: {score_color};">
                            {score}
                        </div>
                        <div style="font-size: 18px; opacity: 0.9;">
                            {grade}
                        </div>
                    </div>
                </div>
            </div>

            <!-- LLM-Generated Coaching Content -->
            <div style="background: #f8f9fa; padding: 25px; border-radius: 10px;
                       border-left: 4px solid #667eea;">
                {html_content}
            </div>

            <!-- Powered by AI Badge -->
            <div style="margin-top: 20px; padding: 15px; background: #e7f3ff;
                       border-radius: 8px; text-align: center; font-size: 13px;
                       color: #004085;">
                <strong>🤖 AI-Enhanced Coaching</strong> • Powered by GPT-4o-mini
                <br>
                <span style="opacity: 0.8;">
                    Combines biomechanical analysis with natural language coaching
                </span>
            </div>
        </div>
        """

        return styled_html

    def _simple_markdown_to_html(self, text: str) -> str:
        """Simple markdown to HTML conversion (fallback)"""
        import re

        # Headers
        text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

        # Lists
        text = re.sub(r'^\- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)

        # Paragraphs
        text = re.sub(r'\n\n', '</p><p>', text)
        text = f'<p>{text}</p>'

        return text

    def _fallback_format(self, feedback_items: List[Dict], score: int) -> str:
        """Fallback formatting if LLM fails"""

        html = f"""
        <div style="background: #fff3cd; border: 2px solid #ffc107;
                   padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <strong>⚠️ AI Enhancement Unavailable</strong><br>
            Showing rule-based feedback (LLM service is currently unavailable)
        </div>

        <h3>Technique Score: {score}/100</h3>

        <h4>Issues Identified:</h4>
        <ul style="line-height: 1.8;">
        """

        for item in feedback_items[:5]:
            severity_icons = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '💡',
                'low': '✨'
            }
            icon = severity_icons.get(item['severity'], '•')

            html += f"""
            <li>
                <strong>{icon} {item['category']}</strong>: {item['issue']}<br>
                <span style="color: #666;">
                    Current: {item['current_value']} | Target: {item['target_range']}
                </span><br>
                <em>{item['recommendation']}</em>
            </li>
            """

        html += "</ul>"
        return html


# Example usage
if __name__ == "__main__":
    # Test LLM enhancer
    enhancer = LLMCoachingEnhancer()

    # Sample feedback data
    sample_feedback = [
        {
            'category': 'Arm Extension',
            'issue': 'Insufficient reach at contact',
            'current_value': '0.72',
            'target_range': '0.82 - 0.95',
            'severity': 'high',
            'recommendation': 'Extend arm fully at contact point'
        },
        {
            'category': 'Power Generation',
            'issue': 'Low swing velocity',
            'current_value': '3.2 m/s',
            'target_range': '4.5 - 5.5 m/s',
            'severity': 'high',
            'recommendation': 'Increase wrist snap speed'
        }
    ]

    enhanced = enhancer.enhance_feedback(
        feedback_items=sample_feedback,
        stroke_type='Clear',
        overall_score=68,
        player_level='intermediate'
    )

    print(enhanced)
