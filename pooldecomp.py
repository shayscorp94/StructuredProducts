import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import relativedelta
from typing import List
from mortgage import Mortgage, FRM, ARM
from dateutil import relativedelta


class MtgPool:
    def __init__(self, mortgage: FRM, num_loans: int, svc_rate: float, g_fee: float, cpr_curve: List[float]):
        self.mortgage = mortgage
        self.num_loans = num_loans
        self.svc_rate = svc_rate
        self.g_fee = g_fee
        self.starting_outstanding_balance = self.mortgage.total_starting_balance * self.num_loans
        self.orig_bal = orig_bal
        self.smm = cpr_curve

    def cf_breakdown(self):
        outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow =\
            np.zeros(self.mortgage.term), np.zeros(self.mortgage.term), np.zeros(self.mortgage.term), np.zeros(self.mortgage.term),np.zeros(self.mortgage.term),np.zeros(self.mortgage.term), np.zeros(self.mortgage.term)
        rate = self.mortgage.note_rate
        term = self.mortgage.term
        pt_rate = rate - self.svc_rate - self.g_fee
        pmt, _, _ = self.mortgage.mtm_payments_generator()
        outstanding_bal[0] = self.starting_outstanding_balance
        mortgage_pmt[0] = pmt*self.num_loans
        for i in range(term):
            if i > 0:
                outstanding_bal[i] = outstanding_bal[i - 1] - total_principal[i - 1]
                mortgage_pmt[i] = FRM(term=term - i, total_starting_balance=outstanding_bal[i], note_rate=rate,
                                        start_date=(self.mortgage.start_date + relativedelta.relativedelta(months=i))
                                      .replace(day=1)).mtm_payments_generator()[0]
            net_interest[i] = outstanding_bal[i] * pt_rate / 12
            scheduled_principal[i] = mortgage_pmt[i] - rate * outstanding_bal[i]/12
            prepayment[i] = self.smm[i] * (outstanding_bal[i] - scheduled_principal[i])
            total_principal[i] = scheduled_principal[i] + prepayment[i]
            cash_flow[i] = net_interest[i] + total_principal[i]
        return outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow


def psa_to_smm(term, psa=100):
    cpr = psa/100 * np.array([.06*t/30 if t < 30 else 0.06 for t in range(1, term + 1)])
    smm = 1 - (1 - cpr)**(1/12)
    return smm


if __name__ == "__main__":
    mortgage = FRM(term=358, total_starting_balance=400000, note_rate=0.06,
                    start_date=datetime(2020, 1, 1))
    num_loans = 1000
    svc_rate = 0.0025
    g_fee = 0.0025
    orig_bal = 100000
    cpr_curve = psa_to_smm(360, psa=165)
    pool = MtgPool(mortgage, num_loans, svc_rate, g_fee, cpr_curve[2:])
    outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow = \
        pool.cf_breakdown()
    print(outstanding_bal)
