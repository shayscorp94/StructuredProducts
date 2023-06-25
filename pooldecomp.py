import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
from mortgage import FRM
from dateutil import relativedelta


class PassThroughPool:
    def __init__(self, mortgage: FRM, num_loans: int, svc_rate: float, g_fee: float, cpr_curve: List[float]):
        """
        Class that defines a pass through (PT) pool/RMBS structure

        Attributes
        ----------
        mortgage: FRM
            FRM object containing terms of a constituent FRM mortgage
        num_loans: int
            Number of loans in the pool
        svc_rate: float
            servicing rate
        g_fee: float
            g_fee imposed by agencies for guaranteeing mortgages, input 0 in case of non-agency pools
        cpr_curve: List[float]
            pre-defined cpr_curve based on market estimations

        """
        self.mortgage = mortgage
        self.num_loans = num_loans
        self.svc_rate = svc_rate
        self.g_fee = g_fee
        self.starting_outstanding_balance = self.mortgage.total_starting_balance * self.num_loans
        self.pt_rate = self.mortgage.note_rate - self.svc_rate - self.g_fee
        self.smm = cpr_curve

    def cf_breakdown(self):
        """
        function that computes scheduled and actual principal balance, interest, and cash-flows resulting from the pool
        based on cpr assumptions
        """
        outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow =\
            np.zeros(self.mortgage.term), np.zeros(self.mortgage.term), np.zeros(self.mortgage.term), \
            np.zeros(self.mortgage.term),np.zeros(self.mortgage.term), np.zeros(self.mortgage.term), \
            np.zeros(self.mortgage.term)
        rate = self.mortgage.note_rate
        term = self.mortgage.term
        pmt, _, _ = self.mortgage.mtm_payments_generator()
        outstanding_bal[0] = self.starting_outstanding_balance
        mortgage_pmt[0] = pmt*self.num_loans
        for i in range(term):
            if i > 0:
                outstanding_bal[i] = outstanding_bal[i - 1] - total_principal[i - 1]
                mortgage_pmt[i] = FRM(term=term - i, total_starting_balance=outstanding_bal[i], note_rate=rate,
                                        start_date=(self.mortgage.start_date + relativedelta.relativedelta(months=i))
                                      .replace(day=1)).mtm_payments_generator()[0]
            net_interest[i] = outstanding_bal[i] * self.pt_rate / 12
            scheduled_principal[i] = mortgage_pmt[i] - rate * outstanding_bal[i]/12
            prepayment[i] = self.smm[i] * (outstanding_bal[i] - scheduled_principal[i])
            total_principal[i] = scheduled_principal[i] + prepayment[i]
            cash_flow[i] = net_interest[i] + total_principal[i]
        return outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow


def psa_to_smm(term, psa=100):
    """
    function to return smm for a given psa (default = 100)
    """
    cpr = psa/100 * np.array([.06*t/30 if t < 30 else 0.06 for t in range(1, term + 1)])
    smm = 1 - (1 - cpr)**(1/12)
    return smm


def wal_calculator(principal_cf: np.array, periods_per_year: int):
    """
    function to calculate WAL for a given set of principal cash-flows
    """
    wal = 0
    for i, cf in enumerate(principal_cf):
        wal += (i + 1)*principal_cf[i]
    return wal/periods_per_year/np.sum(principal_cf)


def pt_pool_table_generator(start_date: datetime, term: int, outstanding_bal: float, mortgage_pmt: float, net_interest: float,
                            scheduled_principal: float, prepayment: float, total_principal: float, cash_flow: float):
    """
    function to print out a time-series dataframe of relevant pool variables
    """
    first_pay_date = (start_date + relativedelta.relativedelta(months=1)).replace(day=1)
    payment_dates_array = np.zeros(term).astype(datetime)
    payment_dates_array[0] = first_pay_date
    for i in range(1, term):
        payment_dates_array[i] = (payment_dates_array[i - 1] + relativedelta.relativedelta(months=1)).replace(day=1)
    df_payments = pd.DataFrame({'Date': payment_dates_array, 'outstanding_bal': outstanding_bal, 'mortgage_pmt': mortgage_pmt,
                                'net_interest': net_interest, 'scheduled_principal': scheduled_principal, 'prepayments': prepayment,
                                'total_principal': total_principal, 'cash_flow': cash_flow})
    return df_payments


if __name__ == "__main__":
    term = 358
    start_date = datetime(2020, 1, 1)
    mortgage = FRM(term=term, total_starting_balance=400000, note_rate=0.06,
                    start_date=start_date)
    num_loans = 1000
    svc_rate = 0.0025
    g_fee = 0.0025
    orig_bal = 100000
    cpr_curve = psa_to_smm(360, psa=165)
    pool = PassThroughPool(mortgage, num_loans, svc_rate, g_fee, cpr_curve[2:])
    outstanding_bal, mortgage_pmt, net_interest, scheduled_principal, prepayment, total_principal, cash_flow = \
        pool.cf_breakdown()
    print(pt_pool_table_generator(start_date=datetime(2020, 1, 1), term=358,
                                  outstanding_bal=outstanding_bal, mortgage_pmt=mortgage_pmt, net_interest=net_interest,
                                  scheduled_principal=scheduled_principal, prepayment=prepayment,
                                  total_principal=total_principal, cash_flow=cash_flow))
