#! python
import sys
import csv
from decimal import Decimal, ROUND_HALF_UP

reader = csv.reader(sys.stdin)
fstr = [row for row in reader]
fmtx = [[sv for sv in row] for row in fstr]
rmtx = [[Decimal(sv).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) \
         for sv in row] for row in fmtx]
print("Your data is read as")
writer = csv.writer(sys.stdout)
writer.writerows(rmtx)
