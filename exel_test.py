import xlrd

book = xlrd.open_workbook('data_book.xlsx')
sheet = book.sheet_by_index(0)
print (sheet.cell(5,0).value)
print (sheet.cell(5,1))
print (5*0)