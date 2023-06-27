import pandas as pd
import numpy as np
from pooldecomp import PassThroughPool, psa_to_smm
from mortgage import FRM
from datetime import datetime
from dateutil import relativedelta


class PACSupportTranche:
    def __init__(self, pac_lower_band: int, pac_upper_band: int, pac_par_amt: float, support_par_amt: float,
                 pt_pool: PassThroughPool, psa: int):
        """
        Class that defines a Planned Amortization Class structure

        Attributes
        ----------
        pac_lower_band: int
            Lower bounds of PSA for the initial PAC collar
        pac_upper_band: int
            Upper bounds of PSA for the initial PAC collar
        pac_par_amt: float
            par amount of PAC tranche
        support_par_amt: float
            par amount of support tranche
        pt_pool: PassThroughPool
            PassThroughPool object composed of collateral of a pool of FRM loans
        psa: float
            psa of underlying cashflows
        """
        self.pac_lower_band = pac_lower_band
        self.pac_upper_band = pac_upper_band
        self.pac_par_amt = pac_par_amt
        self.support_par_amt = support_par_amt
        self.pt_pool = pt_pool
        self.psa = psa

    def cf_breakdown(self):
        """
        function that breaks down pac and support tranche principal and interest payments over time
        """
        outstanding_bal, _, net_interest, _, _, total_principal, _ = self.pt_pool.cf_breakdown()
        mortgage, num_loans, svc_rate, g_fee = self.pt_pool.mortgage, self.pt_pool.num_loans, \
                                               self.pt_pool.svc_rate, self.pt_pool.g_fee
        pt_rate = self.pt_pool.pt_rate
        term_length = mortgage.term
        _, _, _, _, _, pac_lower_principal, _ = PassThroughPool(mortgage, num_loans, svc_rate, g_fee, psa_to_smm(360,
                                                                                                    psa=self.pac_lower_band)[2:]).cf_breakdown()
        _, _, _, _, _, pac_upper_principal, _ = PassThroughPool(mortgage, num_loans, svc_rate, g_fee, psa_to_smm(360,
                                                                                                    psa=self.pac_upper_band)[2:]).cf_breakdown()
        pac_effective_principal = np.minimum(pac_lower_principal, pac_upper_principal)
        outstanding_bal_pac, outstanding_bal_support, pac_actual_principal, pac_actual_interest, support_principal, support_interest = \
            np.zeros(term_length), np.zeros(term_length), np.zeros(term_length), np.zeros(term_length), np.zeros(term_length), np.zeros(term_length)
        outstanding_bal_pac[0] = self.pac_par_amt
        outstanding_bal_support[0] = self.support_par_amt
        # psa within PAC collar
        if self.pac_lower_band <= self.psa <= self.pac_upper_band:
            for i in range(term_length):
                if i > 0:
                    outstanding_bal_pac[i] = outstanding_bal_pac[i - 1] - pac_actual_principal[i - 1]
                pac_actual_principal[i] = pac_effective_principal[i]
                pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i]/12
                support_principal[i] = total_principal[i] - pac_actual_principal[i]
                if i > 0:
                    outstanding_bal_support[i] = max(0, outstanding_bal_support[i - 1] - support_principal[i - 1])
                support_interest[i] = pt_rate * outstanding_bal_support[i]/12
        # psa exceeds PAC collar upper band
        elif self.psa > self.pac_upper_band:
            i = 0
            while i < term_length:
                if i > 0:
                    outstanding_bal_pac[i] = outstanding_bal_pac[i - 1] - pac_actual_principal[i - 1]
                pac_actual_principal[i] = pac_effective_principal[i]
                pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i]/12
                if i > 0:
                    if total_principal[i] - pac_actual_principal[i] > outstanding_bal_support[i - 1] - \
                            support_principal[i - 1]:
                        support_principal[i] = outstanding_bal_support[i]
                        support_interest[i] = pt_rate * outstanding_bal_support[i]/12
                        outstanding_bal_pac[i] = outstanding_bal[i] - outstanding_bal_support[i]
                        pac_actual_principal[i] = total_principal[i] - support_principal[i]
                        pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i]/12
                        i += 1
                        break
                else:
                    if total_principal[i] - pac_actual_principal[i] > outstanding_bal_support[i]:
                        support_principal[i] = outstanding_bal_support[i]
                        support_interest[i] = pt_rate * outstanding_bal_support[i]/12
                        outstanding_bal_pac[i] = outstanding_bal[i] - outstanding_bal_support[i]
                        pac_actual_principal[i] = total_principal[i] - support_principal[i]
                        pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i]/12
                        i += 1
                        break
                support_principal[i] = total_principal[i] - pac_actual_principal[i]
                if i > 0:
                    outstanding_bal_support[i] = max(0, outstanding_bal_support[i - 1] - support_principal[i - 1])
                support_interest[i] = pt_rate * outstanding_bal_support[i]/12
                i += 1
            outstanding_bal_pac[i:] = outstanding_bal[i:]
            pac_actual_principal[i:] = total_principal[i:]
            pac_actual_interest[i:] = net_interest[i:]
        # psa below PAC collar lower band
        else:
            i = 0
            while i < term_length:
                if i > 0:
                    outstanding_bal_pac[i] = outstanding_bal_pac[i - 1] - pac_actual_principal[i - 1]
                if total_principal[i] < pac_effective_principal[i] or \
                        outstanding_bal_pac[i] - np.sum(pac_effective_principal[i:]) > 1e-4:
                    pac_actual_principal[i] = total_principal[i]
                    pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i]/12
                    if i > 0:
                        outstanding_bal_support[i] = outstanding_bal_support[i - 1]
                    support_principal[i] = 0
                    support_interest[i] = pt_rate * outstanding_bal_support[i]/12
                    i += 1
                    continue
                pac_actual_principal[i] = pac_effective_principal[i]
                pac_actual_interest[i] = pt_rate * outstanding_bal_pac[i] / 12
                support_principal[i] = total_principal[i] - pac_actual_principal[i]
                if i > 0:
                    outstanding_bal_support[i] = max(0, outstanding_bal_support[i - 1] - support_principal[i - 1])
                if (outstanding_bal_pac[i] <= np.sum(pac_effective_principal[i:])) \
                        and (outstanding_bal_pac[i - 1] > np.sum(pac_effective_principal[i - 1:])):
                    outstanding_bal_pac[i] = np.sum(pac_effective_principal[i:])
                    outstanding_bal_support[i] = max(0, outstanding_bal[i] - outstanding_bal_pac[i])
                support_interest[i] = pt_rate * outstanding_bal_support[i]/12
                i += 1
        return outstanding_bal_pac, outstanding_bal_support, pac_actual_principal, pac_actual_interest, \
               support_principal, support_interest


def pac_tranche_table_generator(start_date: datetime, term: int, pac_actual_principal: np.ndarray,
                                pac_actual_interest: np.ndarray, support_principal: np.ndarray,
                                support_interest: np.ndarray):
    """
    function to print out a time-series dataframe of balance, principal payments and interest payments for different
    tranches
    """
    first_pay_date = (start_date + relativedelta.relativedelta(months=1)).replace(day=1)
    payment_dates_array = np.zeros(term).astype(datetime)
    payment_dates_array[0] = first_pay_date
    for i in range(1, term):
        payment_dates_array[i] = (payment_dates_array[i - 1] + relativedelta.relativedelta(months=1)).replace(day=1)
    df = pd.DataFrame({'Date': payment_dates_array, 'pac_principal': pac_actual_principal, 'pac_interest': pac_actual_interest,
                       'support_principal': support_principal, 'support_interest': support_interest})
    return df


if __name__ == '__main__':
    term = 358
    start_date = datetime(2020, 1, 1)
    mortgage = FRM(term=term, total_starting_balance=400000, note_rate=0.06,
                    start_date=start_date)
    loan_psa = 165
    num_loans = 1000
    svc_rate = 0.0025
    g_fee = 0.0025
    orig_bal = 100000
    cpr_curve = psa_to_smm(360, psa=loan_psa)
    pt_pool = PassThroughPool(mortgage, num_loans, svc_rate, g_fee, cpr_curve[2:])
    pac_support_tranche = PACSupportTranche(pac_lower_band=100, pac_upper_band=250, pac_par_amt=284984594,
                                           support_par_amt=115015406,
                 pt_pool=pt_pool, psa=loan_psa)
    outstanding_bal_pac, outstanding_bal_support, pac_actual_principal, pac_actual_interest, \
    support_principal, support_interest = pac_support_tranche.cf_breakdown()
    df = pac_tranche_table_generator(start_date, term,
                                             pac_actual_principal, pac_actual_interest, support_principal, support_interest)



