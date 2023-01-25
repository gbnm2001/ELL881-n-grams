import time
import re
import contractions

def removePageNumberTitle(data):
    page = 'Page \| [0-9]+[ \n]+'
    author = 'J.K. Rowling'
    titles = ['<personname> and the philosophers stone',
        '<personname> and the goblet of fire',
        '<personname> and the chamber of secrets',
        '<personname> and the prisoner of azkaban',
        '<personname> and the order of the phoenix',
        '<personname> and the half Blood prince',
        '<personname> and the deathly hallows']
    for t in titles:
        ex = 'page ?\| ?[0-9]+ ?'+t.lower()+' ?- ?j\.k\. rowling'
        print()
        data = re.sub(ex,' ', data)
    data = re.sub('  +', ' ', data)
    return data

def format(input_path, output_path):
    file = open(input_path, encoding="utf8")
    data = file.read()
    data = re.sub('\n',' ', data)
    data = re.sub('  +',' ', data)
# def parseMrMrs():#2
#     global data
    data = re.sub('Mr. ', 'Mr ',data)
    data = re.sub('Mrs. ', 'Mrs ',data)

# def replaceName():#3
#     global data
    full_names = 'Regulus Arcturus Black|Sirius Black|Lavender Brown|Cho Chang|Vincent Crabbe Sr.|Vincent Crabbe|Bartemius Crouch Sr.\
        |Bartemius Crouch Jr.|Bartemius Crouch|Fleur Delacour|Cedric Diggory|Alberforth Dumbledore|Albus Dumbledore|Dudley Dursley\
        |Petunia Dursley|Vernon Dursley|Argus Filch|Seamus Finnigan|Nicolas Flamel|Cornelius Fudge|Goyle Sr.|Goyle|Gregory Goyle\
        |Hermione Granger|Rubeus Hagrid|Igor Karkaroff|Viktor Krum|Bellatrix Lestrange|Alice Longbottom|Frank Longbottom|Neville Longbottom\
        |Luna Lovegood|Xenophilius Lovegood|Remus Lupin|Draco Malfoy|Lucius Malfoy|Narcissa Malfoy|Olympe Maxime|Minerva McGonagall\
        |Alastor Moody|Peter Pettigrew|Harry Potter|James Potter|Lily Potter|Quirinus Quirrell|Tom Riddle Sr.|Tom Riddle|Mary Riddle\
        |Lord Voldemort|Rita Skeeter|Severus Snape|Nymphadora Tonks|Dolores Janes Umbridge|Arthur Weasley|Bill Weasley|Charlie Weasley\
        |Fred Weasley|George Weasley|Ginny Weasley|Molly Weasley|Percy Weasley|Ron Weasley|Dobby|Fluffy|Hedwig|Moaning Myrtle|Aragog|Grawp|Barty\
        |[yY]ou\- ?\n?[kK]now\- ?\n?[wW]ho|He\- ?\n?Who\- ?\n?Must\- ?\n?Not\- ?\n?Be\- ?\n?Named'
    data = re.sub(full_names,'<personname>', data)
    half_names ='Regulus|Arcturus|Black|Sirius|Lavender|Brown|Cho|Chang|Vincent|Crabbe|Bartemius|Crouch|Fleur|Delacour|Cedric|Diggory|Alberforth\
        |Dumbledore|Albus|Dudley|Dursley|Petunia|Vernon|Argus|Filch|Seamus|Finnigan|Nicolas|Flamel|Cornelius|Fudge|Goyle|Gregory|Hermione|Granger\
        |Rubeus|Hagrid|Igor|Karkaroff|Viktor|Krum|Bellatrix|Lestrange|Alice|Longbottom|Frank|Neville|Luna|Lovegood|Xenophilius|Remus|Lupin|Draco\
        |Malfoy|Lucius|Narcissa|Olympe|Maxime|Minerva|McGonagall|Alastor|Mad-Eye|Moody|Peter|Pettigrew|Harry|Potter|James|Lily|Quirinus|Quirrell\
        |Tom|Riddle|Mary|Lord|Voldemort|Rita|Skeeter|Severus|Snape|Nymphadora|Tonks|Dolores|Janes|Umbridge|Arthur|Weasley|Bill|Charlie|Fred|George\
        |Ginny|Molly|Percy|Ron|Dobby|Fluffy|Hedwig|Moaning|Myrtle|Aragog|Grawp|Barty'
    
    data = re.sub(half_names, '<personname>',data)

    data = data.lower()
    data = removePageNumberTitle(data)
    # fc = open('texp.txt', 'w+',encoding='utf8')
    # fc.write(data)
    # fc.close()
    data = re.sub("i'd",'i <d>',data)#'d can be anything
    data = re.sub("yeh", "you", data)
    List = data.split()
    res = []
    for word in List:
        res.append(contractions.fix(word))
    data = ' '.join(res)
    data = re.sub("'s", " <s>", data)#after removal Harry's -> Harry <s>


    data = re.sub('''!|"|#|$|%|&|'|\(|\)|\*|\+|— |/|:|;|=|@|\[|\]|^|_|`|\{|\||\}|~|”|“|\.\.\.|\. \. \.|’|,''','',data)
    data = re.sub('- |—',' ',data)
    data = data.replace('\\', '')
    

    
    data = re.sub('\n\n+','\n',data)
    sentence_list = re.split('\?|\.', data)
    fo = open(output_path, 'w+',encoding='utf-8')
    
    for sentence in sentence_list:
       fo.write(sentence+'\n')
    fo.close()


def merge(filepaths,out):
    files = [open(f,'r',encoding = 'utf8') for f in filepaths]
    data = ''
    for i in files:
        data +=i.read()+'\n'
    fo = open(out, 'w+', encoding='utf8')
    fo.write(data)
    fo.close()
# removeTitleAndPageNumber()
# #print(data[:100])
# parseMrMrs()
# replaceName()
# replaceShortForms()
# removePunctuations()

# data = data.lower()
# sentenceSegmentation()'Harry_Potter_Text/Book1.txt'
books = [
    'Harry_Potter_Text/Book1.txt',
    'Harry_Potter_Text/Book2.txt',
    'Harry_Potter_Text/Book3.txt',
    'Harry_Potter_Text/Book4.txt',
    'Harry_Potter_Text/Book5.txt',
    'Harry_Potter_Text/Book6.txt'
]
merge(books, 'book.txt')
format('book.txt', 'parsed.txt')