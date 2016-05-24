import model

lang = model.Lang()

print(lang.train_data)

lang.train(iterator=200)
print(lang.gen("ABC"))
print(lang.gen("AB"))
print(lang.gen("DEF"))
print(lang.gen("CDEF"))

