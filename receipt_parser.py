from models import Receipt
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError
import json
import re


parser = PydanticOutputParser(pydantic_object=Receipt)

prompt = PromptTemplate(
    template="""You are a strict receipt and transaction parser. Your only job is to extract structured data from messy OCR text.

Extract the following fields:
- "receiver": store name, merchant, or person receiving payment (string or null)
- "date": transaction date in ISO 8601 format YYYY-MM-DD (string or null)
- "total_amount": final amount paid as a number, no currency symbol (float or null)
- "currency": detected currency code e.g. "MYR", "USD" (string or null)
- "category": one of ["groceries", "restaurant", "transport", "utilities", "shopping", "personal_transaction", "healthcare", "entertainment", "other"]
- "confidence": your overall confidence in the extraction, one of ["high", "medium", "low"]

Category classification rules and hints:

* "groceries": 
    - Sells raw, packaged, or uncooked items you bring home
    - Look for: supermarket, hypermarket, convenience store names
    - Items like: rice, eggs, milk, vegetables, canned goods, cleaning products
    - Examples: Giant, Tesco, Aeon, 99 Speedmart, Mydin, KK Mart
    - NOT groceries if items are prepared/cooked on site

* "restaurant":
    - Sells prepared, ready-to-eat, or cooked food
    - Look for: cafe, restaurant, bistro, mamak, warung, food court
    - Items like: nasi lemak, teh tarik, burger, pizza, coffee drinks
    - Examples: McDonalds, Starbucks, Old Town, any mamak stall
    - Even if it sells drinks only (bubble tea, juice bar) → restaurant

* "transport":
    - Related to moving from place to place
    - Look for: Grab, MRT, LRT, bus, taxi, toll, parking, petrol
    - Examples: Petronas, Shell, Touch n Go toll, Rapid KL

* "utilities":
    - Recurring bills or essential services
    - Look for: TNB, Syabas, Unifi, Maxis, Celcom, Digi, Astro
    - Items like: electricity bill, water bill, phone bill, internet

* "shopping":
    - Non-food retail purchases
    - Look for: clothing stores, electronics, department stores
    - Examples: Zara, H&M, Harvey Norman, Parkson, Mr DIY

* "personal_transaction":
    - Money movement between people or accounts
    - Look for: DuitNow, bank transfer, e-wallet, TNG reload
    - Examples: Maybank, CIMB, Touch n Go eWallet top up, PayNow

* "healthcare":
    - Medical or health-related purchases
    - Look for: clinic, hospital, pharmacy, guardian, watson
    - Items like: medicine, consultation fee, supplements
    - Examples: Guardian, Watsons, KPJ, Pantai Hospital

* "entertainment":
    - Leisure and recreation spending
    - Look for: cinema, streaming, games, sports, events
    - Examples: GSC, TGV, Netflix, Steam, Spotify, bowling

* "other":
    - Use this ONLY if none of the above clearly fits
    - When in doubt between two categories, pick the more specific one
    - Do not default to this too quickly

Overall Rules:
- Return ONLY a single valid JSON object — no explanation, no markdown, no code fences
- If a value is uncertain or missing, use null — never fabricate or guess amounts
- Normalize corrupted text (e.g. "T0TAL" -> "TOTAL", "Mlk 2%" -> "Milk 2%")
- For total_amount: prefer the FINAL total (after tax/discount), not subtotal
- For receiver: use the most prominent name at the top of the receipt
- If multiple totals exist, pick the largest one that is labeled as total/grand total

OCR Text:
###
{ocr_text}
###
""",
    input_variables=["ocr_text"]
)

llm = OllamaLLM(model="mistral")

chain = prompt | llm

def extract_json(text: str) -> dict:
    """Try to extract valid JSON from LLM raw output"""
    # Strip markdown code fences if LLM ignores instructions
    text = re.sub(r"```json|```", "", text).strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try finding JSON object within the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"No valid JSON found in output: {text}")

def parse_receipt(ocr_text: str, retries: int = 2) -> Receipt:
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            # Get raw LLM output
            raw_output = chain.invoke({"ocr_text": ocr_text})
            print(f"Attempt {attempt + 1} raw output:\n{raw_output}\n")
            
            # Extract and validate JSON
            json_data = extract_json(raw_output)
            
            # Validate against Pydantic schema
            result = Receipt(**json_data)
            return result
        
        except ValueError as e:
            print(f"Attempt {attempt + 1} - JSON extraction failed: {e}")
            last_error = e
        
        except ValidationError as e:
            print(f"Attempt {attempt + 1} - Schema validation failed: {e}")
            last_error = e
        
        except Exception as e:
            print(f"Attempt {attempt + 1} - Unexpected error: {e}")
            last_error = e
    
    # All retries failed, return safe fallback
    print(f"All {retries + 1} attempts failed. Returning fallback. Last error: {last_error}")
    return Receipt(
        receiver=None,
        date=None,
        total_amount=None,
        currency=None,
        category="other",
        confidence="low"
    )