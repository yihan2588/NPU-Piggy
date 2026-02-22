import time
from database import get_all_receipts, init_db, insert_receipt
from receipt_parser import parse_receipt

# Initialize DB on startup
init_db()


# Mock OCR text to simulate what easyOCR would return
mock_ocr = """
Give us feedback @ survey.walmart.con
Thank you! ID #: 7RDLFKV1D2Y
Walmart >,
ハ
716-646-0682 Mar: KAL.I
5360 SOUTHWESTERN BL.VD
HAMBURG NY
14075
ST# 02405 OP# 009050 TEM 50 TR# 02960
EO NICO LOZ
068113114797
32.98 N
6.88 0
SONOMA PEPP
0830415009997
UDIS GLUTEN
069899781033 F
9.18 0
SUBTOTAL.
49.04
TOTAL
49.04
DEBIT
TEND
49.04
CHANGE DUE
0.00
EFT DEBIT
PAY FROM PRIMARY
49.04
TOTAL PURCHASE
Debit
*x发* 文实发* *次** 8240 I0
REF # 203300709814
NETWORK ID. 0069 APPR CODE 451538
Debit
AID A0000000042203
AAC 650AEFOCB8B71ABC
TERMINAL # SC011046
02/02/22
18:43:02
# ITEMS SOLD 3
TCH 1302 9164 0550 3444 063
Walmartt
Give them the
of mimbership
Scanto gift today,
02/02/22
8:43:08
"""

start = time.time()
result = parse_receipt(mock_ocr)

# Dynamically load OCR text from file for testing
# with open("ocr_input.txt", "r") as f:
#     ocr_text = f.read()

# result = parse_receipt(ocr_text)

end = time.time()

print(result.model_dump())
print(f"\nInference time: {end - start:.2f} seconds")

# Save to DB
insert_receipt(result)

# Verify it was saved
print("\nAll receipts in DB:")
for row in get_all_receipts():
    print(row)