#!/usr/bin/env python
# coding: utf-8

import random
import math
import matplotlib.pyplot as plt
import argparse
import sys
import pickle
import numpy as np
from pdb import set_trace
import statistics

# import copy

random.seed(11)


class Config:
    def __init__(self):
        self.T = 1000  # time (1000)
        self.N = 1000  # number of firms
        self.Ñ = 180  # size parameter

        # Φ = 0.1  # capital productivity (constant and uniform)
        self.c = 1  # parameter bankruptcy cost equation
        self.α = 0.08  # alpha, ratio equity-loan
        self.g = 1.1  # variable cost
        self.w = 0.005  # wage
        self.ω = 0.002  # markdown interest rate (the higher it is, the monopolistic power of banks)
        self.λ = 0.0  # credit assets rate
        self.d = 100  # location cost
        self.e = 0.1  # sensivity
        self.β = 0.02  # beta
        self.σ = 0.0  # R&D sigma
        self.k = 1  # capital intensity
        self.η = 0.25  # market regime (0.25 = monopoly ; 0.0001 = perfect competition)
        self.b = 1
        self.δ = 2
        self.ζ = 0.003
        self.alpha_r = 0.5
        self.share_K = 0.5
        self.ans_cash = 1
        self.share_cash = 0.03
        self.fire_sale = 0.88

        # firms initial parameters
        self.K_i0 = 5  # capital
        self.A_i0 = 1  # asset
        self.L_i0 = 4  # liability
        self.π_i0 = 0  # profit
        self.B_i0 = 0  # bad debt

        # risk coefficient for bank sector (Basel)
        self.v = 0.08


Config = Config()


# %%
class Statistics:
    doLog = False  # imposta doLog su false, ma se mettiamo --log diventa True

    def log(cadena):  # sta creando un metodo "log(con un argomento)
        if Statistics.doLog:  # se abbiamo messo --log e quindi doLog sta su True, allora:
            print(cadena)  # stampa quello che sta come argomento della funzione log

    firms = []
    bankSector = []

    bankrupcy = []
    firmsK = []
    firmsπ = []
    firmsL = []
    firmsB = []
    rate = []
    list_a = [Config.A_i0]
    list_k_med = []
    list_a_med = []
    list_l_med = []

    outputK = []
    outputY = []
    profits = []
    phis = []

    current_total_phi = 0.0

    removedPhi = []

    phi_per_impresa = []

    def getStatistics(self):  # sta creando un metodo "getStatistics"
        global args  # Viene dichiarata la variabile globale args, che si presume sia stata definita altrove nel codice. La variabile args viene utilizzata per accedere agli argomenti passati al programma da riga di comando
        # quando viene chiamata getStatistics() in doSimulation richiama il metodo log(cadena) della classe Statistics() e printa il suo argomento (quindi quello che ho messo tra parentesi)

        Statistics.log(
            f"σ = {Config.σ} - η = {Config.η} - t = {Status.t:4d} [firms] n = {len(Status.firms)}, sumA = {Status.firmsAsum:.2f}, sumL = {Status.firmsLsum:.2f}, sumK = {Status.firmsKsum:.2f}, sumπ = {Status.firmsπsum:.2f}")

        Statistics.log(
            f"[bank]  avgRate= {BankSector.getAverageRate(self):.4f}, D = {BankSector.D:.2f}, L = {BankSector.L:.2f}, E = {BankSector.E:0.2f}, B = {BankSector.B:.2f}, π = {BankSector.π:.2f}")
        ##Statistics.log( " r=%s " % Status.firms[0].r )
        Statistics.firmsK.append(Status.firmsKsum)  # aggiunge alla lista firmsK i risultati di firmsKsum in Status
        Statistics.firmsπ.append(Status.firmsπsum)  # aggiunge alla lista i profitti che si trovano nella classe Status
        Statistics.firmsL.append(
            Status.firmsLsum)  # aggiunge alla lista firmsL i loans che si trovano nella classe Status
        Statistics.firmsB.append(
            BankSector.B)  # aggiunge alla lista firmsB il bad debt calcolato nella classe BankSector e che viene richiamato nella funzione doSimulation
        Statistics.rate.append(BankSector.getAverageRate(
            self))  # aggiunge alla lista rate il tasso calcolato nella classe BankSector che poi da in return il risultato

        if args.saveall:
            bank = {}
            bank[
                'L'] = BankSector.L  # crea un dict "bank" con chiave L e prende il valore dalla classe BankSector, dove ha la funzione updateBankL() che viene richiamata in doSimulation()
            bank[
                'D'] = BankSector.D  # crea una chiave D nel dict "bank" e prende il valore dalla classe BankSector dove c'è BankSector.D che viene aggiornato quando viene richiamata in doSimulation() la funzione updateBankSector()
            bank['avgrate'] = BankSector.getAverageRate(
                self)  # viene usato il valore nella funzione definita nella classe BankSector
            bank[
                'E'] = BankSector.E  # aggiunge la chiave E (equity) tramite la classe BankSector dove è definito un valore iniziale di E, ma che poi viene aggiornato con la funzione determineEquity() richiamata in updateBankSector() a sua volta richiamata da doSimulation()
            bank[
                'D'] = BankSector.D  # aggiunge la chiave D al dict "bank" usando il valore di D in BankSector che viene calcolato dalla funzione determineDeposits(), poi viene assegnato in updateBankSector() che viene richiamato in doSimulation()
            bank[
                'π'] = BankSector.π  # aggiunge la chiave π al dict "bank" che viene calcolata nella classe BankSector() tramite la funzione determineProfits() che poi lo assegna in updateBankSector() che viene richiamato in doSimulation()
            firms = []  # crea una lista di imprese
            for i in Status.firms:  # ciclo for da 1 fino al numero di imprese indicato nella classe Status
                firm = {}
                firm['K'] = i.K
                firm['r'] = i.r
                firm['L'] = i.L
                firm['π'] = i.π
                firm['u'] = i.u
                firms.append(firm)
            Statistics.firms.append(firms)
            Statistics.bankSector.append(bank)

    @staticmethod
    def inizializzazione(prova):
        prova.firms = []
        prova.bankSector = []

        prova.bankrupcy = []
        prova.firmsK = []
        prova.firmsπ = []
        prova.firmsL = []
        prova.firmsB = []
        prova.rate = []
        prova.list_a = [Config.A_i0]
        prova.list_k_med = []
        prova.list_a_med = []
        prova.list_l_med = []
        prova.removedPhi = []
        prova.current_total_phi = 0.0
        prova.phi_per_impresa = []


class Status:
    firms = []  # lista che viene aggiornata con la funzione initialiize richiamata a sua volta da doSimulation()
    firmsKsum = 0.0  # questi saranno aggiornati con la funzione updatefirmsStatus() di sotto
    firmsAsum = 0.0  # sarà aggiornato con la funzione updatefirmsStatus()
    firmsLsum = 0.0  # sarà aggiornato con la funzione updatefirmsStatus()
    firmsπsum = 0.0  # sarà aggiornato con la funzione updatefirms()
    numFailuresGlobal = 0
    t = 0

    firmsKsums = []  # crea una lista contentente la somma dei K totali
    firmsGrowRate = []  # crea una lista contenente i tassi di crescita ( k al tempo t - k al tempo t-1 / k al tempo t-1)

    firmIdMax = 0  # genera il numero massimo di id corrispondenti alle imprese nel settore

    list_total_l = []
    list_total_a = []
    min_equity = 0.0
    median_capital = 0.0
    median_equity = 0.0
    median_loans = 0.0

    mode_capital = 0.0

    current_period = 0.0

    sommaK = 0.0
    lista_sommaK = []
    lista_profitti = []
    lista_phis = []
    lista_sommaY = []

    def getNewFirmId(self):
        Status.firmIdMax += 1
        return Status.firmIdMax

    @staticmethod
    def initialize():
        for i in range(Config.N):
            Status.firms.append(
                Firm())  # aggiunge alla lista firms creata prima le caratteristiche delle imprese specificate nella classe Firms()

    @staticmethod
    def inizializzaizone(prova):
        prova.firms = []  # lista che viene aggiornata con la funzione initialiize richiamata a sua volta da doSimulation()
        prova.firmsKsum = 0.0  # questi saranno aggiornati con la funzione updatefirmsStatus() di sotto
        prova.firmsAsum = 0.0  # sarà aggiornato con la funzione updatefirmsStatus()
        prova.firmsLsum = 0.0  # sarà aggiornato con la funzione updatefirmsStatus()
        prova.firmsπsum = 0.0  # sarà aggiornato con la funzione updatefirms()
        prova.numFailuresGlobal = 0
        prova.t = 0

        prova.firmsKsums = []  # crea una lista contentente la somma dei K totali
        prova.firmsGrowRate = []  # crea una lista contenente i tassi di crescita ( k al tempo t - k al tempo t-1 / k al tempo t-1)

        prova.firmIdMax = 0  # genera il numero massimo di id corrispondenti alle imprese nel settore

        prova.list_total_l = []
        prova.list_total_a = []
        prova.min_equity = 0.0
        prova.median_capital = 0.0
        prova.median_equity = 0.0
        prova.median_loans = 0.0
        prova.mode_capital = 0.0

        prova.current_period = 0.0


class Firm():
    # K = Config.K_i0  # capital
    K = 0.0
    # A = Config.A_i0  # asset
    A = 0.0
    r = 0.0  # rate money is given by banksector
    # L = Config.L_i0  # credit
    L = 0.0
    Credit_supply = None
    C = 0.0  # cash
    π = 0.0  # profit
    u = 0.0  # shock generato da un'uniforme tra 0 e 2
    γ = (Config.ω / Config.k) + Config.g * r
    Φ = 0.10  # capital productivity
    Y = Φ * K
    Z = 0.0
    z = None
    list_k = [Config.K_i0]
    I = None
    success = 0.0
    dK = 0.0
    past_losses = 0.0
    cash = 0.0
    previous_phi = Φ

    def __init__(self):
        self.id = Status.getNewFirmId(self)  # metodo costruttore dove pone un id di ogni firm uguale a quello che genera nella classe Status mediante la funzione getNewFirmId()
        if Status.current_period == 0:
            self.K = Config.K_i0
            self.A = Config.A_i0
            self.L = Config.L_i0
        else:
            self.K = Status.median_capital
            self.A = Status.median_equity
            self.L = Status.median_loans

        if self.A + self.L - self.K > 0:
            self.A = self.K - self.L
        else:
            self.K = self.A + self.L

    def determineInvestment(self):
        return self.dK - self.K  # inserire una lista che tiene conto dei vari K delle imprese

    def determineCreditSupply(self):
        return (Config.λ * BankSector.L * (self.K / Status.firmsKsum)) + \
            ((1 - Config.λ) * BankSector.L * (self.A / Status.firmsAsum))

    def determineCredit(self):  # determina il credito che le imprese vogliono sulla base dell'equazione 11
        # (equation 11)                     # se negativo poni A = A di ieri + abs(L di oggi) e L di oggi = 0
        return self.L + self.I - self.π

    def determineInterestRate(self):  # determina il tasso di interesse usando equazione 12
        # (equation 32)
        return Config.β * (self.L / self.A)  # da inserire minimo 0.01 e massimo 0.1 ??

    # siccome uso i valori di ieri posso calcolarlo subito
    # anche prima di gamma

    def __ratioK(self):  # determina il ratioK dell'impresa facendo il rapporto tra il suo K e quello totale
        return self.K / Status.firmsKsum

    def __ratioA(self):  # determina gli asset relativi dell'impresa dividendo i suoi asset per gli asset totali
        return self.A / Status.firmsAsum

    def determineGamma(self):
        return ((Config.w / Config.k) + Config.g * self.r)

    def determineCapital(self):  # determina il K desiderato tramite l'equazione 9
        # equation 28
        # se è negativo, bisognerebbe imporlo pari a K_i0/2 e bisogna imporre un controllo: se < di share_K*K di ieri
        return (1 - Config.σ) * (((1 - Config.η) ** 2 * (1 - Config.σ) * self.Φ - (1 - Config.η) * self.γ) / (
                    Config.b * self.Φ * self.γ)) + (self.A / (2 * self.γ))

    # def determineCash(self):  # dopo aver calcolate il capitale dovremmo devolvere una parte al cash
    # return Config.share_cash * self.K  # il K che abbiamo calcolato prima è il K operativo

    def determineOutput(self):
        return self.Φ * self.K

    def determineU(self):  # genera la u, lo shock
        return random.uniform(0, 2)

    def determineAssets(self):  # genera l'evoluzione dell'equity dell'impresa
        # equation 7
        return self.A + self.π  # K - self.L

    def determineProfit(self):  # determina i profitti dell'impresa dall'equazione 5
        # equation 23
        return (self.u * (Config.η + (1 - Config.η) * self.Φ * self.K) * (1 - Config.σ)) - (self.γ * self.K)

    def determine_z(self):
        # equazione 23
        return (self.u * (Config.η + (1 - Config.η) * self.Φ * self.K) * Config.σ) / self.K

    def determineZ(self):
        # equazione 29
        # return 1 - math.exp(- Config.δ * (self.z))
        return 1 - np.exp(- Config.δ * (self.z))

    def determineSuccess(self):
        return np.random.binomial(1, self.Z)

    def determinePhi(self):
        if self.success == 1:
            self.Φ = (1 + Config.ζ) * self.Φ
        return self.Φ

    @staticmethod
    def inizializzazione(prova):
        prova.K = 0.0
        prova.A = 0.0
        prova.r = 0.0  # rate money is given by banksector
        prova.L = 0.0
        prova.Credit_supply = None
        prova.C = 0.0  # cash
        prova.π = 0.0  # profit
        prova.u = 0.0  # shock generato da un'uniforme tra 0 e 2
        prova.γ = (Config.ω / Config.k) + Config.g * prova.r
        prova.Φ = 0.10  # capital productivity
        prova.Y = prova.Φ * prova.K
        prova.Z = 0.0
        prova.z = None
        prova.list_k = [Config.K_i0]
        prova.I = None
        prova.success = 0.0
        prova.dK = 0.0
        prova.past_losses = 0.0
        prova.cash = 0.0
        prova.previous_phi = 0.1


class BankSector():  # crea la classe riferita alla singola banca
    E = Config.N * Config.L_i0 * Config.v
    B = Config.B_i0  # bad debt
    D = 0
    π = 0
    π_list = []

    def determineDeposits(self):  # determina i depositi, uguali ai loan concessi - equity della Banca
        # as a residual from L = E+D, ergo D=L-E
        return BankSector.L - BankSector.E  # controllare se < 0 e imporli a 0

    def determineProfit(self):  # determina i profitti della banca
        # equation 13
        profitDeposits = 0.0
        for firm in Status.firms:
            profitDeposits += firm.r * firm.L  # determina i profitti sui crediti concessi come tasso * loans concessi
        BankSector.D = BankSector.determineDeposits(self)  # detemrina l'ammontare dei depositi
        resto = (BankSector.getAverageRate(self) * (((1 - Config.ω) * BankSector.D) + BankSector.E))
        # resto = (0.01 * (((1 - Config.ω) * BankSector.D) + BankSector.E))

        ###Statistics.log("        - bank profit= dep(%s) - %s , which  %s * [(1-w)*%s+%s]"%( profitDeposits  ,resto, BankSector.getAverageRate(), BankSector.D , BankSector.E ))
        # total_profit = profitDeposits - (BankSector.getAverageRate(self) * ((1 - Config.ω) * (BankSector.D + BankSector.E)))  # profitti = interessi sui crediti concessi  - tasso * (depositi + equity ... quindi interessi sui depositi dei correntisti e sull'equity posseduta dagli azionisti)
        total_profit = profitDeposits - resto
        return total_profit

    def getAverageRate(self):
        average = 0.0
        sumL = 0.0
        for firm in Status.firms:
            average += firm.r  # firm.r sarebbe il tasso di interesse applicato all'impresa, che si determina con determineInterestRate() richiamata da updateFirms() a sua volta richiamata da doSimulation()
        return average / len(Status.firms)  # tutti i tassi di interesse che sono presenti in average / il totale delle firm... restituisce il tasso medio
        # average += firm.r * firm.L
        # sumL += firm.L
        # return average / sumL

    def determineEquity(self):  # determina equity della banca
        # equation 14
        # print(f"{BankSector.π} + {BankSector.E} - {BankSector.B}")
        result = BankSector.π + BankSector.E - BankSector.B  # equity = profitti + equity che ho immagazzinata al tempo precedente - bad debt
        return result

    @staticmethod
    def inizializzazione(prova):
        prova.E = Config.N * Config.L_i0 * Config.v
        prova.B = Config.B_i0  # bad debt
        prova.D = 0
        prova.π = 0
        prova.π_list = []


def removeBankruptedFirms():
    i = 0
    BankSector.B = 0.0
    for firm in Status.firms[:]:
        # if (firm.π + firm.A) < Status.min_equity:  # se profitto + equity impresa sono negative... l'impresa fallisce
        if firm.A < 0.3:
            # print(f"K = {firm.K}, A = {firm.A} e L = {firm.L}")
            if firm.L < (Config.fire_sale * firm.K):
                BankSector.B += 0
            else:
                BankSector.B += (firm.L - (Config.fire_sale * firm.K))  # **********************************             #il bad debt della banca aumenta di una somma pari al credito concesso all'impresa - il suo capitale
                # print(f"{firm.L} - {Config.fire_sale*firm.K}")
            Status.firms.remove(firm)  # rimuovi dalla lista firms nella classe Status() l'impresa
            Status.numFailuresGlobal += 1  # aggiungi il contatore di imprese fallite
            i += 1  # aggiorna il numero di imprese fallite
            Statistics.removedPhi.append(firm.Φ)
    Statistics.log("        - removed %d firms %s" % (i, "" if i == 0 else " (next step B=%s)" % BankSector.B))
    Statistics.bankrupcy.append(i)  # aggiunge alla lista bankruptcy nella classe Statistics() il numero di imprese fallite che aggiorna con la i
    return i


"""
def addFirms(Nentry):
    for i in range(Nentry):
        Status.firms.append(Firm())  # aggiunge alla lista firms della classe Status, l'impresa i-esima
    Statistics.log("        - add %d new firms (Nentry)" % Nentry)
    #print (f"I valori mediani sono: K = {Status.median_capital}, A = {Status.median_equity},"
           #f" L = {Status.median_loans}")
"""


def addFirms(Nentry):
    # tot_phi = 0.0
    avg_phi = 0.0
    modal_phi = 0.0
    # print(Statistics.current_total_phi)
    avg_phi = Statistics.current_total_phi / 1000
    Statistics.current_total_phi = 0.0
    #if len(Statistics.phi_per_impresa) > 0:  # Controlla se la lista non è vuota
        #modal_phi = statistics.mode(Statistics.phi_per_impresa)
        #print(f"phi modale = {modal_phi:.4f}")
    #else:
        #print("ciaooo")
        #modal_phi = 0.1  # Imposta un valore di default per modal_phi se la lista è vuota
    print(f"il phi medio è:{avg_phi}")
    Statistics.phi_per_impresa.clear()
    for i in range(Nentry):
        new_firm = Firm()  # Crea una nuova impresa con attributi di default
        #new_firm.Φ = avg_phi  # Assegna a phi il valore medio
        if Config.σ == 0.0:
            new_firm.Φ = avg_phi  # Assegna a phi il valore medio
        else:
            new_firm.Φ = random.uniform(avg_phi - 0.0015, avg_phi + 0.0015) # provo a inserire una perturbazione
        # print(f"L'impresa {i} è rientrata con {new_firm.Φ}")
        # print(avg_phi)
        Status.firms.append(new_firm)  # Aggiunge alla lista firms della classe Status, l'impresa i-esima
    Statistics.log("        - add %d new firms (Nentry)" % Nentry)


def updateFirmsStatus():
    Status.firmsAsum = 0.0
    Status.firmsKsum = 0.0
    Status.firmsLsum = 0.0
    for firm in Status.firms:
        Status.firmsAsum += firm.A  # sarà aggiornato con la funzione updateFirmsStatus() che sarà richiamata da updateFirms() che sarà a sua volta richiamata in doSimulation()
        Status.firmsKsum += firm.K  # sarà calcolato richiamando la funzione updateFirms() che poi sarà richiamata da doSimulation()
        Status.firmsLsum += firm.L  # sarà aggiornato richiamando al funzione updateFirms() che poi sarà richiamata da doSimulation()

    Status.firmsKsums.append(
        Status.firmsKsum)  # aggiunge alla lista firmsKsums nella classe Status(), la somma dei K calcolati sopra, derivanti da updateFirms()
    Status.firmsGrowRate.append(
        0 if Status.t == 0 else (Status.firmsKsums[Status.t] - Status.firmsKsums[Status.t - 1]) / Status.firmsKsums[
            Status.t - 1])  # calcola tasso di crescita e lo aggiunge alla lista firmsGrowthRate

    """
    if Status.t == 0:
        Status.firmsGrowRate.append(0)
    elif Status.t < len(Status.firmsKsums):
        Status.firmsGrowRate.append((Status.firmsKsums[Status.t] - Status.firmsKsums[Status.t - 1]) / Status.firmsKsums[Status.t - 1])
    """


def updateFirms():  # sarà richiamata in doSimulation()
    # update Kt-1 and At-1 (Status.firmsKsum && Status.firmsAsum):
    # updateFirmsStatus()                       #???????????????????????? secondo me dovrebbe stare alla fine
    totalK = 0.0
    totalL = 0.0
    totalA = 0.0
    totalπ = 0.0
    totalphis = 0.0
    totalY = 0.0
    averagephis = 0.0
    Status.firmsπsum = 0.0
    i = 0
    n = 0
    for firm in Status.firms:
        # updateBankL()  # aggiorna i loans della banca ogni periodo
        firm.Credit_supply = firm.determineCreditSupply()
        firm.r = firm.determineInterestRate()
        # if firm.r <= 0.03:
        # firm.r = 0.03
        if firm.r >= 0.10:
            firm.r = 0.10
        firm.γ = firm.determineGamma()  # secondo me calcolo prima il tasso di interesse perché usa i valori di ieri
        firm.dK = firm.determineCapital()
        if firm.dK < Config.share_K * firm.K:
            firm.dK = Config.share_K * firm.K
        if firm.dK < 0:
            firm.dK = Config.K_i0 / 2
        firm.I = firm.determineInvestment()
        firm.L = firm.determineCredit()  # determina i loans con la funzione determineCredit()
        if firm.L < 0:
            firm.A = firm.A + abs(firm.L)
            firm.L = 0
        if firm.A + firm.L - firm.dK > 0:
            firm.A = firm.dK - firm.L
        if firm.Credit_supply >= firm.L:
            firm.K = firm.dK
        else:
            firm.L = firm.Credit_supply
            firm.K = firm.L + firm.A
        if firm.L < 0:
            firm.A = firm.A + abs(firm.L)
            firm.L = 0
        if firm.K < 0:
            print("K è negativo")
        #firm.Y = firm.determineOutput()
        firm.u = firm.determineU()
        firm.π = firm.determineProfit()
        Status.firmsπsum += firm.π
        firm.A = firm.determineAssets()
        # print(f"{firm.K} - {firm.A} e {firm.L}")
        if firm.A + firm.L - firm.K > 0:
            firm.A = firm.K - firm.L
        else:
            firm.K = firm.A + firm.L
        firm.Y = firm.determineOutput()
        totalK += firm.K
        totalL += firm.L
        totalA += firm.A
        totalπ += firm.π
        totalY += firm.Y
        Statistics.list_a.append(firm.A)
        Statistics.list_k_med.append(firm.K)
        Statistics.list_a_med.append(firm.A)
        Statistics.list_l_med.append(firm.L)
        firm.z = firm.determine_z()
        firm.Z = firm.determineZ()
        firm.success = firm.determineSuccess()
        # print(f"firm{firm.id}: {firm.success}")
        firm.previous_phi = firm.Φ
        firm.Φ = firm.determinePhi()
        # print(f"firm{firm.id}:  {firm.Φ}")
        totalphis += firm.Φ
        i += 1
        Statistics.current_total_phi += firm.Φ
        Statistics.phi_per_impresa.append(firm.Φ)
    averagephis = totalphis / len(Status.firms)
    Status.list_total_l.append(totalL)
    Status.list_total_a.append(totalA)
    Status.lista_sommaK.append(totalK)
    Status.lista_profitti.append(totalπ)
    Status.lista_phis.append(averagephis)
    Status.lista_sommaY.append(totalY)
    updateFirmsStatus()
    Status.min_equity = min(Statistics.list_a)
    #print(f"lunghezza = {len(Statistics.list_k_med)}")
    #print(f"K medio = {Status.median_capital}")
    Status.median_capital = statistics.mean(Statistics.list_k_med) # se metti statistics.mode funziona, ma con la media sballa
    #Status.mode_capital = statistics.mean(Statistics.list_k_med)
    #for i in range(len(Statistics.list_k_med)):
        #if Statistics.list_k_med[i] == Status.median_capital:
            #n += 1
    #print(f"il valore modale è ripetuto {n} volte ed è pari a {Status.mode_capital}")
    if Status.median_capital < 1.5:
        Status.median_capital = 1.5
    Status.median_equity = statistics.mean(Statistics.list_a_med)   #se metti la moda va, altrimenti sballa
    if Status.median_equity < 0.5:
        Status.median_equity = 0.5
    Status.median_loans = statistics.mean(Statistics.list_l_med)    # se metti la moda va, sennò sballa
    if Status.median_loans < 1:
        Status.median_loans = 1
    if Status.median_equity + Status.median_loans > Status.median_capital:
        Status.median_equity = Status.median_capital - Status.median_loans
    else:
        Status.median_capital = Status.median_equity + Status.median_loans
    print(f"K medio = {Status.median_capital}, A medio = {Status.median_equity} e L medio = {Status.median_loans}")
    Statistics.list_a.clear()
    Statistics.list_k_med.clear()
    Statistics.list_a_med.clear()
    Statistics.list_l_med.clear()


def determineNentry(self):  # determina il numero di nuovi entranti che poi sarà usato inella funzione addFirms()
    # equation 15
    # return round(Config.Ñ / (1 + math.exp(Config.d * (BankSector.getAverageRate(self) - Config.e))))
    last_period_bankrupt_firms = Statistics.bankrupcy[-1] if Statistics.bankrupcy else 0
    return last_period_bankrupt_firms


def updateBankL():  # determina i loans della banca che può concedere in base al coefficiente di Basilea
    BankSector.L = BankSector.E / Config.v


def updateBankSector():
    totalL = 0.0
    for firm in Status.firms:
        totalL += firm.L
    BankSector.D = totalL - BankSector.E
    if BankSector.D < 0:
        BankSector.D = 0
    BankSector.π = BankSector.determineProfit(BankSector)  # determina il profitto della banca e lo aggiorna
    BankSector.π_list.append(BankSector.π)
    # print(f"E di ieri : {BankSector.E}, Profitti : {BankSector.π} e Bad Debt : {BankSector.B}")
    BankSector.E = BankSector.determineEquity(BankSector)  # determina l'equity della banca e l'aggiorna
    # BankSector.D = BankSector.L - BankSector.E  # determina i depositi della banca e li aggiorna


def doSimulation(doDebug=False):
    Status.initialize()  # inizia la simulazione e richiama la funzione initialize()
    Status.lista_sommaK.append(5000)
    Status.lista_profitti.append(0)
    Status.lista_phis.append(0.1)
    Status.lista_sommaY.append(500)
    updateFirmsStatus()  # richiama la funzione che aggirona lo stato delle imprese nella loro totalità (infatti usa Ksums, Asums, Lsums)
    updateBankL()  # richiama la funzione che determina i loans della banca
    BankSector.D = BankSector.L - BankSector.E  # determina i depositi della banca
    Status.current_period = 0
    for t in range(Config.T):
        if t == 0:
            print("sono nell'if")
            print(f"pre - sumA = {Status.firmsAsum}, sumL = {Status.firmsLsum}, sumK = {Status.firmsKsum}")
            Status.inizializzaizone(Status)
            BankSector.inizializzazione(BankSector)
            Firm.inizializzazione(Firm)
            Statistics.inizializzazione(Statistics)
            print(f"sumA = {Status.firmsAsum}, sumL = {Status.firmsLsum}, sumK = {Status.firmsKsum}")
            Status.initialize()
            updateFirmsStatus()  # richiama la funzione che aggirona lo stato delle imprese nella loro totalità (infatti usa Ksums, Asums, Lsums)
            updateBankL()  # richiama la funzione che determina i loans della banca
            BankSector.D = BankSector.L - BankSector.E  # determina i depositi della banca
            Status.current_period = 0
        Status.t = t  # imposta la variabile t nella classe Status al tempo t che va da 1 a 1000
        Statistics.getStatistics(Statistics)  # richiama la funzione getStatistics() e fa vedere le varie statistiche con un print
        #removeBankruptedFirms()  # richiama la funzione per rimuovere le imprese in bancarotta
        #newFirmsNumber = determineNentry(BankSector)  # associa alla variabile newFirmsNumber il risultato della funzione determineNentry
        #addFirms(newFirmsNumber)  # richiama la funzione addFirms e come argomento gli passa il risultato di newFirmsNumber
        updateBankL()
        updateFirms()  # aggiorna i risultati delle imprese ogni periodo e richiama anche updateFirmsStatus()
        updateBankSector()  # aggiorna i risultati delle funzioni della banca ogni periodo e richiama anche BankSector.D
        removeBankruptedFirms()
        newFirmsNumber = determineNentry(BankSector)
        addFirms(newFirmsNumber)
        Status.current_period += 1

        if doDebug and (doDebug == t or doDebug == -1):
            set_trace()  # DOVREBBE essere una funzione di debug

    Statistics.outputK.append(Status.lista_sommaK)
    Status.lista_sommaK = []
    Statistics.profits.append(Status.lista_profitti)
    Status.lista_profitti = []
    Statistics.phis.append(Status.lista_phis)
    Status.lista_phis = []
    Statistics.outputY.append(Status.lista_sommaY)
    Status.lista_sommaY = []


def graph_zipf_density(show=True):  # staremo creando i presupposti per una distribuzione
    Statistics.log("zipf_density")  # fa uscire una stringa zipf_density
    plt.clf()  # puliamo il grafico
    zipf = {}  # log K = freq      #crea un dizionario che tiene conto dei diversi valori di K (arrotondati) sui quali verrà calcolato il logaritmo naturale
    for firm in Status.firms:
        if round(firm.K) > 0:  # se il valore arrotondato di K è maggiore di 0, allora
            x = math.log(round(firm.K))  # calcola il suo logaritmo
            if x in zipf:
                zipf[
                    x] += 1  # se il risultato da noi ottenuto è presente nel dict, allora a quella chiave associa un valore +1
            else:
                zipf[
                    x] = 1  # se il valore da noi ottenuto non è presente, crea una chiave con quel valore e dalle valore di 1
    x = []  # crea una lista x
    y = []  # crea una lista y
    for i in zipf:
        x.append(
            i)  # aggiunge il valore i-esimo di zipf alla lista x (asse ascisse)... quindi vede le varie chiavi (che sono i log del K) e le aggiunge alla lista x
        y.append(
            math.log(zipf[i]))  # agggiunge il logaritmo della frequenza per quel valore specifico di log(round (K) )
    plt.plot(x, y, 'o', color="blue")  # "o" = i punti sono cerchi
    plt.ylabel("log freq")  # etichetta asse ordinate
    plt.xlabel("log K")  # etichetta asse ascisse
    plt.title("Zipf plot of firm sizes")  # titolo grafico
    plt.show() if show else plt.savefig("zipf_density.svg")  # show se show == True oppure salvalo


def graph_zipf_density1(show=True):
    Statistics.log("zipf_density")  # fammi uscire una stringa con scritto "zipf_density"
    plt.clf()  # pulisci il grafico
    zipf = {}  # log K = freq           #crea un dict "zipf"
    for firm in Status.firms:
        if round(firm.K) > 0:  # se il valore arrotondato di firm.K è maggiore di 0, allora:
            x = math.log(round(firm.K))  # calcola il log di K arrotondato
            if x in zipf:
                zipf[
                    x] += 1  # se quel valore sta già, aggiungi 1 al suo valore (è una coppia chiave-valore, dove chiave = log(round(K)) e valore = contatore)
            else:
                zipf[x] = 1  # inserisci una nuova chiave con quel log(round(K)) e metti 1 al suo valore (contatore)
    x = []
    y = []
    for i in zipf:
        if math.log(zipf[i]) >= 1:  # se il log delle frequenze è > 1, allora:
            x.append(i)  # aggiungi il valore della i-esima chiave in zipf alla lista x
            y.append(math.log(zipf[i]))  # aggiungi alla lista y il log delle frequenze
    plt.plot(x, y, 'o', color="blue")
    plt.ylabel("log freq")
    plt.xlabel("log K")
    plt.title("Zipf plot of firm sizes (modified)")
    plt.show() if show else plt.savefig("zipf_density1.svg")


def graph_zipf_rank(
        show=True):  # questa funzione crea un grafico Zipf che mostra il rango dei valori K rispetto ai loro logaritmi. Un grafico Zipf di questo tipo può essere utile per visualizzare la distribuzione di valori che seguono una legge di potenza
    Statistics.log("zipf_rank")  # produci una riga che scriva zipf_rank
    plt.clf()  # pulisci il grafico
    y = []  # log K = freq          #inserisci il log di K
    x = []
    for firm in Status.firms:
        if round(firm.K) > 0:  # se il valore arrotondato di K è maggiore di 0, allora:
            y.append(math.log(firm.K))  # inserisci il log di firm.K
    y.sort();  # ordina la lista y
    y.reverse()  # falla diventare decrescente...il valore log(round(K)) più grande sarà il primo elemento di y, e il più piccolo sarà l'ultimo
    for i in range(len(y)):  # ciclo che itera fino all'ultimo valore di y
        x.append(math.log(float(i + 1)))  # aggiungi alla lista x il log di (i + 1)... perché su python si inizia da 0
    plt.plot(y, x, 'o', color="blue")
    plt.xlabel("log K")
    plt.ylabel("log rank")
    plt.title("Rank of K (zipf)")
    plt.show() if show else plt.savefig("zipf_rank.svg")


def graph_aggregate_output(show=True):
    Statistics.log("aggregate_output")  # fa vedere una stringa "aggregate output"
    plt.clf()  # pulisce il grafico
    xx1 = []  # crea una lista "xx1"
    yy = []  # crea una lista "yy"
    for i in range(150, Config.T):  # itera per i che va da 150 a 1000 (Config.T è uguale a 1000)
        yy.append(i)  # mette nella lista yy, il valore di i che parte da 150 e arriva a 1000
        xx1.append(math.log(Status.firmsKsums[
                                i]))  # aggiunge alla lista xx1 il log della somma dei K delle varie imprese dal tempo 150 al tempo 1000 e vede come si evolve
    plt.plot(yy, xx1,
             'b-')  # yy rappresentano le ascisse (il tempo) e xx1 i logdella somma dei capitali, mentre "b-" i valori devono essere tracciati con una linea
    plt.ylabel("log K")  # asse delle y denominata log K
    plt.xlabel("t")  # asse delle x denominata t
    plt.title("Logarithm of aggregate output")
    plt.show() if show else plt.savefig(f"aggregate_output_sigma{Config.σ}.svg")


def graph_profits(show=True):
    Statistics.log("profits")
    plt.clf()
    xx = []  # crea una lista "xx"
    yy = []  # crea una lista "yy"
    for i in range(150, Config.T):  # itera per i che va da 150 a 1000 (Config.T è 1000)
        xx.append(i)  # aggiungi le varie i (che rappresentano gli stanti temporali)
        yy.append(Statistics.firmsπ[
                      i] / Config.N)  # aggiungi la somma dei proditti delle imprese al tempo t e dividili per le imprese presenti nell'economia al tempo t
    plt.plot(xx, yy, 'b-')
    plt.ylabel("avg profits")
    plt.xlabel("t")
    plt.title("profits of companies")  # determina i profitti medi delle imprese nel tempo
    plt.show() if show else plt.savefig("profits.svg")


def graph_baddebt(show=True):
    Statistics.log("bad_debt")  # stringa "bad_debt"
    plt.clf()  # pulisci il grafico
    xx = []
    yy = []
    for i in range(0, Config.T):
        xx.append(i)  # inserisci gli istanti temporali nella lista xx
        yy.append(Statistics.firmsB[
                      i] / Config.N)  # inserisci nella lista yy le statistiche inerente al bad debt (con il segno "-" perché un debito) totale delle imprese presente nella classe Statistics() e dividi per il numero di imprese
    plt.plot(xx, yy, 'b-')
    plt.ylabel("avg bad debt")  # bad debt medio
    plt.xlabel("t")
    plt.title("Bad debt")
    plt.show() if show else plt.savefig("bad_debt_avg.svg")


def graph_bankrupcies(show=True):
    Statistics.log("bankrupcies")  # stringa che dice "bankruptcies"
    plt.clf()  # pulisci il grafico
    xx = []
    yy = []
    for i in range(150, Config.T):  # prende in considerazione i periodi dal 150-esimo al 1000-esimo
        xx.append(i)
        yy.append(Statistics.bankrupcy[i])  # prende il numero di imprese fallite contenuto nella lista
    plt.plot(xx, yy, 'b-')
    plt.ylabel("num of bankrupcies")
    plt.xlabel("t")
    plt.title("Bankrupted firms")  # numero di bancarotte in ogni tempo t
    plt.show() if show else plt.savefig("bankrupted.svg")


def graph_bad_debt(show=True):
    Statistics.log("bad_debt")
    plt.clf()
    xx = []
    yy = []
    for i in range(150, Config.T):
        if Statistics.firmsB[
            i] < 0:  # se il bad debt delle imprese al tempo t è < 0 (siccome è un debito si...), allora:
            xx.append(i)  # aggiungi quel tempo alla lista "xx"
            yy.append(math.log(-Statistics.firmsB[
                i]))  # aggiungi alla lista yy il log di (-bad debt... questa volta con segno meno perché il bad debt è negativo)
        # else:
        # print("%d %s" % (i, Statistics.firmsB[i]))  # altrimenti se non c'è bad debt negativo, stampa il periodo t e il valore del debito inesigibile
    plt.plot(xx, yy, 'b-')
    plt.ylabel("ln B")
    plt.xlabel("t")
    plt.title("Bad debt")
    plt.show() if show else plt.savefig("bad_debt.svg")


def graph_interest_rate(show):
    Statistics.log("interest_rate")
    plt.clf()
    xx2 = []
    yy = []
    for i in range(150, Config.T):
        yy.append(i)  # aggiungi alla lista "yy" il valore i-esimo del tempo (che parte da 150 e arriva a 1000)
        xx2.append(Statistics.rate[
                       i])  # aggiungi il rate indicato nella classe Statistics.rate[al tempo t] che viene aggiornata nella funzione getStatistics() che viene richiamatada doSimulation()
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("mean rate")
    plt.xlabel("t")
    plt.title("Mean interest rates of companies")
    plt.show() if show else plt.savefig("interest_rate.svg")


def graph_growth_rate(show):
    Statistics.log("growth_rate")
    plt.clf()
    xx2 = []
    yy = []
    for i in range(150, Config.T):
        if Status.firmsGrowRate[i] != 0:  # se il growth rate è diverso da 0
            yy.append(i)  # inserisci il valore i-esimo del tempo
            xx2.append(Status.firmsGrowRate[i])  # inserisci nella lista "xx2" il growthrate al tempo t
    plt.plot(yy, xx2, 'b-')
    plt.ylabel("growth")
    plt.xlabel("t")
    plt.title("Growth rates of agg output")
    plt.show() if show else plt.savefig("growth_rates.svg")


def graph_bank_profits(show=True):
    Statistics.log("Bank profits")
    plt.clf()
    xx = []  # crea una lista "xx"
    yy = []  # crea una lista "yy"
    for i in range(0, Config.T):  # itera per i che va da 150 a 1000 (Config.T è 1000)
        xx.append(i)  # aggiungi le varie i (che rappresentano gli stanti temporali)
        yy.append(BankSector.π_list[
                      i])  # aggiungi la somma dei proditti delle imprese al tempo t e dividili per le imprese presenti nell'economia al tempo t
    plt.plot(xx, yy, 'b-')
    plt.ylabel("bank profits")
    plt.xlabel("t")
    plt.title("profits of the Bank")  # determina i profitti medi delle imprese nel tempo
    plt.show() if show else plt.savefig("bank_profits.svg")


def graph_firm_L(show=True):
    Statistics.log("Loans of the firms")
    plt.clf()
    xx = []  # crea una lista "xx"
    yy = []  # crea una lista "yy"
    for i in range(0, Config.T):  # itera per i che va da 150 a 1000 (Config.T è 1000)
        xx.append(i)  # aggiungi le varie i (che rappresentano gli stanti temporali)
        yy.append(Status.list_total_l[
                      i])  # aggiungi la somma dei proditti delle imprese al tempo t e dividili per le imprese presenti nell'economia al tempo t
    plt.plot(xx, yy, 'b-')
    plt.ylabel("Loans of firms")
    plt.xlabel("t")
    plt.title("Loans of the firms")  # determina i profitti medi delle imprese nel tempo
    plt.show() if show else plt.savefig("Loans of the firms.svg")


def graph_firm_A(show=True):
    Statistics.log("Equity of the firms")
    plt.clf()
    xx = []  # crea una lista "xx"
    yy = []  # crea una lista "yy"
    for i in range(0, Config.T):  # itera per i che va da 150 a 1000 (Config.T è 1000)
        xx.append(i)  # aggiungi le varie i (che rappresentano gli stanti temporali)
        yy.append(Status.list_total_a[
                      i])  # aggiungi la somma dei proditti delle imprese al tempo t e dividili per le imprese presenti nell'economia al tempo t
    plt.plot(xx, yy, 'b-')
    plt.ylabel("Equity of the firms")
    plt.xlabel("t")
    plt.title("Equity of the firms")  # determina i profitti medi delle imprese nel tempo
    plt.show() if show else plt.savefig("Equity of the firms.svg")


def show_graph(show):  # se True mostrami i grafici elencati
    graph_aggregate_output(show)
    graph_growth_rate(show)
    graph_zipf_rank(show)
    graph_zipf_density(show)
    graph_zipf_density1(show)
    graph_profits(show)
    graph_bad_debt(show)
    graph_baddebt(show)
    graph_bankrupcies(show)
    graph_interest_rate(show)
    graph_bank_profits(show)
    graph_firm_L(show)
    graph_firm_A(show)


def save(filename,
         all=False):  # save ( nome del file in cui salvare i dati, all = False, un sottoinsieme di dati viene salvato)
    try:
        with open(filename,
                  'wb') as file:  # with open(filename, 'wb') as file: apre il file specificato in modalità di scrittura binaria ('wb'), e il file viene automaticamente chiuso quando il blocco with è terminato
            if all:  # if all: controlla se il parametro all è True. Se lo è, allora tutti i dati vengono salvati.
                pickle.dump(Statistics.firms,
                            file)  # pickle.dump(Statistics.firms, file) e le linee simili salvano diverse variabili nel file. Queste variabili contengono vari tipi di dati riguardanti le imprese nel modello, come il capitale (K), i profitti (π), i prestiti (L), i debiti inesigibili (B), ecc...
                pickle.dump(Statistics.bankSector, file)
            else:  # Se all non è True, allora solo un sottoinsieme dei dati viene salvato
                pickle.dump(Statistics.firmsK, file)
                pickle.dump(Statistics.firmsπ, file)
                pickle.dump(Statistics.firmsL, file)
                pickle.dump(Statistics.firmsB, file)
                pickle.dump(Status.firms, file)
                pickle.dump(Statistics.bankrupcy, file)
                pickle.dump(Statistics.rate, file)
                pickle.dump(Status.firmsKsums, file)
                pickle.dump(Status.firmsGrowRate, file)
    except Exception:  # cattura tutte le eccezioni che si possono verificare
        print("not possible to save %s to %s" % ("all" if all else "status",
                                                 filename))  # stampa un messaggio di errore se si verifica un'eccezione durante il salvataggio dei dati. Il messaggio include se stava tentando di salvare tutti i dati o solo lo stato, e il nome del file in cui stava cercando di salvare i dati


def restore(filename,
            all=False):  # La funzione restore è l'opposto della funzione save. Mentre save salva i dati su un file, restore legge i dati da un file
    global args
    try:
        with open(filename,
                  'rb') as file:  # with open(filename, 'rb') as file: apre il file specificato in modalità di lettura binaria ('rb'), e il file viene automaticamente chiuso quando il blocco with è terminato
            if all:  # if all: controlla se il parametro all è True. Se lo è, allora tutti i dati vengono caricati. Se non lo è, allora solo un sottoinsieme dei dati viene caricato
                Statistics.firms = pickle.load(
                    file)  # Statistics.firms = pickle.load(file) e le linee simili caricano diverse variabili dal file. Queste variabili contengono vari tipi di dati riguardanti le imprese nel modello, come il capitale (K), i profitti (π), i prestiti (L), i debiti inesigibili (B), ecc...
                Statistics.bankSector = pickle.load(file)
            else:
                Statistics.firmsK = pickle.load(file)
                Statistics.firmsπ = pickle.load(file)
                Statistics.firmsL = pickle.load(file)
                Statistics.firmsB = pickle.load(file)
                Status.firms = pickle.load(file)
                Statistics.bankrupcy = pickle.load(file)
                Statistics.rate = pickle.load(file)
                Status.firmsKsums = pickle.load(file)
                Status.firmsGrowRate = pickle.load(file)
    except Exception:  # cattura qualsiasi eccezione
        print("not possible to restore %s from %s" % ("all" if all else "status",
                                                      filename))  # stampa un messaggio di errore se si verifica un'eccezione durante il caricamento dei dati. Il messaggio include se stava tentando di caricare tutti i dati o solo lo stato, e il nome del file da cui stava cercando di caricare i dati
        sys.exit(
            0)  # sys.exit(0) termina il programma se si verifica un'eccezione durante il caricamento dei dati. Il 0 indica che il programma è terminato normalmente, nonostante l'eccezione

    if not args.savegraph and not args.graph:  # controlla se i flag --savegraph e --graph non sono impostati
        set_trace()  # Se entrambi i flag non sono impostati, viene chiamata la funzione set_trace(), che probabilmente avvia il debugger per consentire l'ispezione interattiva del codice
    else:
        show_graph(
            args.graph)  # Se uno dei due flag è impostato, viene chiamata la funzione show_graph() passando come argomento il valore del flag --graph
    # try:
    #    code.interact(local=locals())
    # except SystemExit:
    #    pass

    # argparse.ArgumentParser() crea un oggetto parser che rappresenta il parser degli argomenti. L'argomento description fornisce una breve descrizione del programma


parser = argparse.ArgumentParser(
    description="Fluctuations firms/banks")  # argparse.ArgumentParser() crea un oggetto parser che rappresenta il parser degli argomenti. L'argomento description fornisce una breve descrizione del programma
parser.add_argument("--graph", action="store_true",
                    help="Shows the graph")  # nelle parentesi indica gli argomenti che il programma che può accettare
parser.add_argument("--sizeparam", type=int, help="Size parameter (default=%s)" % Config.Ñ)
parser.add_argument("--savegraph", action="store_true", help="Save the graph")
parser.add_argument("--log", action="store_true", help="Log to stdout")
parser.add_argument("--debug", help="Do a debug session at t=X, default each t", type=int, const=-1, nargs='?')
parser.add_argument("--saveall", type=str, help="Save all firms data (big file: file will be overwritten)")
parser.add_argument("--restoreall", type=str, help="Restore all firms data (big file: and enters interactive mode)")
parser.add_argument("--save", type=str, help="Save the state (file will be overwritten)")
parser.add_argument("--restore", type=str, help="Restore the state (and enters interactive mode)")
parser.add_argument("--sigma", type=float, help="Value for sigma in the model (default=%s" % Config.δ)

args = parser.parse_args()  # esegue il parsing degli argomenti da riga di comando, restituendo un oggetto args che contiene i valori dei flag e delle opzioni specificate dall'utente

if args.sizeparam:  # Se l'argomento --sizeparam è stato specificato, viene aggiornato il valore di Config.Ñ con il valore fornito dall'utente
    Config.Ñ = int(args.sizeparam)
    if Config.Ñ < 0 or Config.Ñ > Config.N:
        print("value not valid for Ñ: must be 0..%s" % Config.N)

if args.log:  # Se l'argomento --log è stato specificato, viene impostata la variabile Statistics.doLog su True, che abilita il logging
    Statistics.doLog = True

if args.restoreall or args.restore:  # Se è stato specificato l'argomento --restoreall o --restore, viene chiamata la funzione restore per ripristinare i dati dal file specificato
    if args.restoreall:
        restore(args.restoreall, True)
    else:
        restore(args.restore, False)
else:  # Se non sono stati specificati gli argomenti --restoreall o --restore, viene chiamata la funzione doSimulation per eseguire la simulazione principale
    if args.sigma:
        Config.δ = args.sigma
    doSimulation(args.debug)
    if Status.numFailuresGlobal > 0:  # Se Status.numFailuresGlobal è maggiore di zero, viene stampato un messaggio sul numero totale di fallimenti
        Statistics.log("[total failures in all times = %s " % Status.numFailuresGlobal)
    else:
        Statistics.log("[no failures]")  # altrimenti dice "no failures"
    if args.save:  # Se è stato specificato l'argomento --save, viene chiamata la funzione save per salvare lo stato in un file
        save(args.save, False)
    if args.saveall:  # Se è stato specificato l'argomento --saveall, viene chiamata la funzione save per salvare tutti i dati delle imprese in un file
        save(args.saveall, True)
    if args.graph:  # Se è stato specificato l'argomento --graph, viene chiamata la funzione show_graph per visualizzare il grafico corrispondente
        show_graph(True)
    if args.savegraph:  # Se è stato specificato l'argomento --savegraph, viene chiamata la funzione show_graph per salvare il grafico in un file
        show_graph(False)
    
for sigma in Config.sigma_values:
        Config.σ = sigma
       


def comparison_of_K(show=True):
    print(Statistics.outputK)
    plt.clf()

    for sigma_index, sigma in enumerate(Config.sigma_values):
        xx1 = []
        yy = []
        for i in range(150, Config.T):
            yy.append(i)
            xx1.append(math.log(Statistics.outputK[sigma_index][i]))
        plt.plot(yy, xx1, label=f'Sigma={sigma}')

    plt.legend()
    plt.ylabel("log K")
    plt.xlabel("t")
    plt.title("Logarithm of aggregate output")

    plt.show() if show else plt.savefig("aggregate_output.svg")

def comparison_of_K_boxplot(show=True):
    print(Statistics.outputK)
    plt.clf()

    # Estrae le serie di dati da Statistics.outputK a partire dal tempo 150
    data = [[math.log(y) for y in x[150:]] for x in Statistics.outputK]

    # Traccia il boxplot
    plt.boxplot(data, labels=[f'Sigma={sigma}' for sigma in Config.sigma_values])

    plt.ylabel("log K")
    plt.xlabel("Sigma Values")
    plt.title("Boxplot of Logarithm of Aggregate Output")

    plt.show() if show else plt.savefig("aggregate_output_boxplot_comparison.svg")

def comparison_of_profits_of_companies(show=True):
    print(Statistics.profits)
    plt.clf()

    for sigma_index, sigma in enumerate(Config.sigma_values):
        xx1 = []
        yy = []
        for i in range(150, Config.T):
            yy.append(i)
            xx1.append(Statistics.profits[sigma_index][i])
        plt.plot(yy, xx1, label=f'Sigma={sigma}')

    plt.legend()
    plt.ylabel("profitti imprese")
    plt.xlabel("t")
    plt.title("Profitti delle imprese")

    plt.show() if show else plt.savefig("profits of companies.svg")


def comparison_of_phis(show=True):
    # print(f"Length of Status.firms: {len(Status.firms)}")
    # print(f"Length of Statistics.phis: {len(Statistics.phis)}")
    # for i in range(len(Status.firms)):
    # print([x/len(Status.firms) for x in Statistics.phis[i]])
    print(f"{Statistics.phis}")
    plt.clf()

    for sigma_index, sigma in enumerate(Config.sigma_values):
        xx1 = []
        yy = []
        for i in range(0, Config.T):
            yy.append(i)
            # log_phi = np.log(Statistics.phis[sigma_index][i])
            # xx1.append(log_phi)
            xx1.append(math.log(Statistics.phis[sigma_index][i]))
        plt.plot(yy, xx1, label=f'Sigma={sigma}')

    plt.legend()
    plt.ylabel("phi imprese")
    plt.xlabel("t")
    plt.title("Phi delle imprese")

    plt.show() if show else plt.savefig("phis of companies.svg")


def comparison_of_phis_boxplot(show=True):
    # print(f"Length of Status.firms: {len(Status.firms)}")
    # print(f"Length of Statistics.phis: {len(Statistics.phis)}")
    # for i in range(len(Status.firms)):
    # print([x/len(Status.firms) for x in Statistics.phis[i]])
    print(f"{Statistics.phis}")
    plt.clf()

    data = [[math.log(y) for y in x[150:]] for x in Statistics.phis]
    plt.boxplot(data, labels=[f'Sigma={sigma}' for sigma in Config.sigma_values])
    plt.ylabel("log K")
    plt.xlabel("Sigma Values")
    plt.title("Boxplot of Logarithm of Phis")

    plt.show() if show else plt.savefig("aggregate_output_boxplot_comparison.svg")

def comparison_of_Y(show=True):
    print(Statistics.outputY)
    plt.clf()

    for sigma_index, sigma in enumerate(Config.sigma_values):
        xx1 = []
        yy = []
        for i in range(150, Config.T):
            yy.append(i)
            xx1.append(math.log(Statistics.outputY[sigma_index][i]))
        plt.plot(yy, xx1, label=f'Sigma={sigma}')

    plt.legend()
    plt.ylabel("log K")
    plt.xlabel("t")
    plt.title("Logarithm of Y = aggregate output")

    plt.show() if show else plt.savefig("Y_aggregate_output.svg")


def comparison_of_boxplotY(show=True):
    print(Statistics.outputK)
    plt.clf()

    # Estrae le serie di dati da Statistics.outputK a partire dal tempo 150
    data = [[math.log(y) for y in x[150:]] for x in Statistics.outputY]

    # Traccia il boxplot
    plt.boxplot(data, labels=[f'Sigma={sigma}' for sigma in Config.sigma_values])

    plt.ylabel("log K")
    plt.xlabel("Sigma Values")
    plt.title("Boxplot of Logarithm of Aggregate Output Y")

    plt.show() if show else plt.savefig("aggregate_output_boxplot_comparison.svg")



comparison_of_K(True)
comparison_of_K_boxplot(True)
comparison_of_profits_of_companies(True)
comparison_of_phis(True)
comparison_of_phis_boxplot(True)
comparison_of_Y(True)
comparison_of_boxplotY(True)