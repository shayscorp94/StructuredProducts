import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
from mortgage import FRM
from pooldecomp import PassThroughPool, psa_to_smm
from dateutil import relativedelta


class SequentialTranche:
    def __init__(self, pt_pool: PassThroughPool, num_tranches: int, par_amounts: List[float]):
        """
        Class that defines a sequential payment tranching structure

        Attributes
        ----------
        pt_pool: PassThroughPool
            A PassThroughPool object composed of multiple FRM mortgages
        num_tranches: int
            number of tranches in the structure
        par_amounts: List[float]
            starting balances/par amounts of each tranche
        """
        self.pt_pool = pt_pool
        self.num_tranches = num_tranches
        self.par_amounts = par_amounts

    def cf_breakdown(self):
        """
        function that breaks down balances, principal payments and interest payments for each tranche over time
        """
        outstanding_bal, _, net_interest, _, _, total_principal, _ = self.pt_pool.cf_breakdown()
        term_length = len(outstanding_bal)
        pt_rate = self.pt_pool.pt_rate
        balance_array, principal_array, interest_array = np.zeros([self.num_tranches, term_length]), \
                                                         np.zeros([self.num_tranches, term_length]),\
                                                         np.zeros([self.num_tranches, term_length])
        for tranche in range(self.num_tranches):
            balance_array[tranche][0] = self.par_amounts[tranche]
            interest_array[tranche][0] = net_interest[0] * self.par_amounts[tranche] / np.sum(self.par_amounts)
        principal_array[0][0] = total_principal[0]
        i, k = 1, 0
        while k < self.num_tranches:
            while i < term_length:
                balance_array[k][i] = balance_array[k][i - 1] - principal_array[k][i - 1]
                interest_array[k][i] = pt_rate * balance_array[k][i]/12
                if total_principal[i] > balance_array[k][i]:
                    principal_array[k][i] = balance_array[k][i]
                    break
                else:
                    principal_array[k][i] = total_principal[i]
                i += 1
            k += 1
            if k >= self.num_tranches:
                break
            balance_array[k][1: i + 1] = self.par_amounts[k]
            interest_array[k][1: i + 1] = self.par_amounts[k] * pt_rate/12
            principal_array[k][i] = total_principal[i] - principal_array[k - 1][i]
            i += 1
        return balance_array, interest_array, principal_array


def sequential_tranche_table_generator(start_date: datetime, term: int, num_tranches: int,
                                       balance_array: np.ndarray, interest_array: np.ndarray,
                                       principal_array: np.ndarray):
    """
    function to print out a time-series dataframe of balance, principal payments and interest payments for different
    tranches
    """
    full_list = [balance_array, interest_array, principal_array]
    first_pay_date = (start_date + relativedelta.relativedelta(months=1)).replace(day=1)
    payment_dates_array = np.zeros(term).astype(datetime)
    payment_dates_array[0] = first_pay_date
    for i in range(1, term):
        payment_dates_array[i] = (payment_dates_array[i - 1] + relativedelta.relativedelta(months=1)).replace(day=1)
    tranche_list = [chr(65 + i) for i in range(num_tranches)]
    metric_list = ['remaining_balance', 'principal', 'interest']
    header = [np.repeat(tranche_list, len(metric_list)), np.tile(metric_list, len(tranche_list))]
    df = pd.DataFrame(columns=header)
    df['Date'] = payment_dates_array
    for tranche in range(num_tranches):
        for metric in range(len(metric_list)):
            df[tranche_list[tranche], metric_list[metric]] = full_list[metric][tranche][:]
    return df


if __name__ == "__main__":
    start_date = datetime(2020, 1, 1)
    term = 358
    num_tranches = 4
    num_loans = 1000
    svc_rate = 0.0025
    g_fee = 0.0025
    orig_bal = 100000
    cpr_curve = psa_to_smm(360, psa=165)
    mortgage = FRM(term=term, total_starting_balance=400000, note_rate=0.06,
                   start_date=start_date)
    pt_pool = PassThroughPool(mortgage, num_loans, svc_rate, g_fee, cpr_curve[2:])
    tranche = SequentialTranche(pt_pool=pt_pool, num_tranches=num_tranches, par_amounts=[194500000, 36000000, 96500000,
                                                                                         73000000])
    balance_array, interest_array, principal_array = tranche.cf_breakdown()
    print(sequential_tranche_table_generator(datetime(2020, 1, 1), 358, 4,
                                             balance_array, interest_array, principal_array))
