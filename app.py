from flask import Flask , request
from flask import render_template
import pdftotext
import csv
import pandas as pd
import numpy as np
import re
import spacy
from spacy.tokens import DocBin
from spacy import displacy
import json


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/",methods=['POST'])
def predict():
    pdf_input = request.files['pdffile']
    pdf_path = "./factures/" + pdf_input.filename
    pdf_input.save(pdf_path)
    
    model_test =""

    #pdf_input = "/home/mohamed/Téléchargements/factures/39.pdf"

    pdf_input = pdf_input.filename

    text_output = "/home/mohamed/Bureau/flask/factures/" + pdf_input[:-4] + '.txt'

    # output final
    csv_output = "/home/mohamed/Bureau/flask/factures/" + pdf_input[:-4] + '.csv'

    with open("/home/mohamed/Bureau/flask/factures/" + pdf_input, "rb") as f:
        pdf = pdftotext.PDF(f)
        
    with open(text_output, 'w') as f:
    # convertir pdf en fichoer txt
        f.write("\n\n".join(pdf).encode('ascii', 'ignore').decode('ascii'))
    
    with open(text_output, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)

        with open(csv_output, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)
            
    data = pd.read_csv(csv_output, error_bad_lines=False,header=None, sep='\t')

    test = np.savetxt(pdf_input[:-4] + '.txt', data.values, fmt='%s') 

    with open(pdf_input[:-4] + '.txt') as f:
        lines = f.readlines()
    
    def lower(text):
       return [x.lower() for x in text]

    model_test= ''.join(map(str, lines))
    model_test = re.sub(' +', ' ', model_test)

    model_test = lower(model_test)

    #stopwords = []
    #resultwords  = [word for word in model_test if word.lower() not in stopwords]

    result = ''.join(model_test)

    # some text processing
    result = result.replace('bon commande',' bon command\n') 
    result = result.replace('date','\ndate',2).replace('date livraison','\ndate livraison').replace('n commande','\nnum commande').replace('bon command','bon commande').replace('montant','\ntotal').replace('total','\ntotal').replace('/2021','/2021\n').replace('/2022','/2022\n').replace('/2020','/2020\n')

    result = re.sub(r'^$\n','', result, flags=re.MULTILINE)

    #print(result)

    with open(pdf_input[:-4] + '.txt', 'w') as f:
        f = f.write(result) #savefinalfile

   
    with open(pdf_input[:-4] + '.txt') as file:
        data = file.readlines()

    model_test =''    
    model_test= ''.join(map(str, data))
       
    # load the trained model
    nlp_output = spacy.load("/home/mohamed/Documents/newfinal/output/model-best")

    # pass our test instance into the trained pipeline
    doc = nlp_output(model_test)

    # customize the label colors
    colors = {"totalttc": "linear-gradient(90deg, #E1D436, #F59710)","numcommande": "linear-gradient(90deg, #FF94EC, #FF94EC)","tva": "linear-gradient(90deg, #aa9cfc, #F32177)",
            "typefacture": "linear-gradient(90deg, #aa9cfc, #fc9ce7)","commandepar": "linear-gradient(90deg, #aa9cfc, #21F38C)","destinataire": "linear-gradient(90deg ,#aa9cfc, #F3E021)"
              ,"commandea": "linear-gradient(90deg, #33FF4F, #33FF4F)","article": "linear-gradient(90deg, #61ECFC, #61ECFC)","datecommande": "linear-gradient(90deg, #FCFC61, #FCFC61)"
              ,"datelivraison": "linear-gradient(90deg ,#FC8D61, #FC8D61)","totalht": "linear-gradient(90deg, #aa9cfc, #7fffd4)","libelle": "linear-gradient(90deg, #E1D436, #F59710)"
              ,"codearticle": "linear-gradient(90deg, #33FF4F, #33FF4F)","eanprincipal": "linear-gradient(90deg, #61ECFC, #61ECFC)", "nbcolis": "linear-gradient(90deg, #FCFC61, #FCFC61)", "pcb": "linear-gradient(90deg, #F3E021, #F3E021)"
              ,"quantite": "linear-gradient(90deg, #FC8D61, #FC8D61)","prixachat": "linear-gradient(90deg, #aa9cfc, #aa9cfc)","montantpromo": "linear-gradient(90deg, #aa9cfc, #F3E021)", "nbuc": "linear-gradient(90deg, #F59710, #E1D436)"
              ,"pc": "linear-gradient(90deg, #33FF4F, #FC8D61)", "qte": "linear-gradient(90deg, #61ECFC, #FC8D61)", "totalhorstax": "linear-gradient(90deg, #aa9cfc, #FC8D61)", "quantuc": "linear-gradient(90deg, #F59710, #FC8D61)", "unite": "linear-gradient(90deg, #aa9cfc, #E1D436)","ucuvc": "linear-gradient(90deg, #61ECFC, #E1D436)"}
    options = {"ents": ["totalttc","tva","typefacture","commandepar","destinataire","commandea","article","datecommande","datelivraison","totalht","numcommande","libelle","codearticle","nbcolis","pcb","nbuc","quantite","eanprincipal","prixachat","montantpromo","pc","qte","totalhorstax","quantuc","unite","ucuvc"], "colors": colors}

    # visualize the identified entities
    displacy.render(doc, style="ent", options=options)

    # print out the identified entities#
    [(ent.label_, ent.text) for ent in doc.ents] 

    typefacture = {}
    numcommande = {}
    datecommande = {}
    datelivraison = {}
    commandepar = {}
    destinataire = {}
    commandea = {}
    codearticle = []
    libelle = []
    eanprincipal = []
    pc = []
    nbcolis = []
    quantuc = []
    pcb =[]
    nbuc = []
    qte = []
    quantite = []
    ucuvc = []
    unite = []
    prixachat = []
    montantpromo = []
    totalhorstax = []
    tva = []
    article = {'codearticle':codearticle,'eanprincipal':eanprincipal,'libelle':libelle,'pc':pc,'nbcolis':nbcolis,'quantuc':quantuc,'pcb':pcb,'nbuc':nbuc,'qte':qte,'quantite':quantite,'ucuvc':ucuvc,'unite':unite,'prixachat':prixachat,'montantpromo':montantpromo,'totalhorstax':totalhorstax,'tva':tva}
    articles = {'articles': article }

    for ent in doc.ents:
        if ent.label_ == 'typefacture':
            typefacture['typefacture'] = ent.text    
        if ent.label_ == 'numcommande':
            numcommande['numcommande'] = ent.text
        if ent.label_ == 'datecommande':
            datecommande['datecommande'] = ent.text
        if ent.label_ == 'datelivraison':
            datelivraison['datelivraison'] = ent.text
        if ent.label_ == 'commandepar':
            commandepar['commandepar'] = ent.text
        if ent.label_ == 'destinataire':
            destinataire['destinataire'] = ent.text
        if ent.label_ == 'commandea':
            commandea['commandea'] = ent.text
        if ent.label_ == "codearticle":
            codearticle.append(ent.text)
        if ent.label_ == "eanprincipal":
            eanprincipal.append(ent.text)    
        if ent.label_ == "libelle":
            libelle.append(ent.text)
        if ent.label_ == "pc":
            pc.append(ent.text)    
        if ent.label_ == "nbcolis":
            nbcolis.append(ent.text)    
        if ent.label_ == "quantuc":
            quantuc.append(ent.text)
        if ent.label_ == "pcb":
            pcb.append(ent.text)
        if ent.label_ == "nbuc":
            nbuc.append(ent.text)    
        if ent.label_ == "qte":
            qte.append(ent.text)    
        if ent.label_ == "quantite":
            quantite.append(ent.text)    
        if ent.label_ == "ucuvc":
            ucuvc.append(ent.text)
        if ent.label_ == "unite":
            unite.append(ent.text)
        if ent.label_ == "prixachat":
            prixachat.append(ent.text)
        if ent.label_ == "montantpromo":
            montantpromo.append(ent.text)
        if ent.label_ == "totalhorstax":
            totalhorstax.append(ent.text)    
        if ent.label_ == "tva":
            tva.append(ent.text)    

    data =[typefacture,numcommande,datecommande,datelivraison,commandepar,destinataire,commandea,articles]

    print(data)    

    #with open("/home/mohamed/Bureau/flask/factures/facture.json", 'a') as f:
        #json.dump(data,f)
        #f.write('\n')
    
    return render_template('index.html', prediction = data)

if __name__ == '__main__':
    app.run(debug=True)