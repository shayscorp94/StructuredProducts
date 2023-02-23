import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import relativedelta


class Mortgage:
    def __init__(self, term: int, total_starting_balance: float, start_date: datetime, deferred_balance=0):
        self.term = term
        self.total_starting_balance = total_starting_balance
        self.start_date = start_date
        self.deferred_balance = deferred_balance

    pass


class FRM(Mortgage):
    def __init__(self, term: int, total_starting_balance: float, start_date: datetime, note_rate: float,
                 deferred_balance=0):
        super().__init__(term, total_starting_balance, start_date, deferred_balance)
        self.note_rate = note_rate

    def mtm_payments_generator(self):
        p_0 = self.total_starting_balance - self.deferred_balance
        r = self.note_rate
        term = self.term
        c, balance_array, interest_array = frm_term_scheduler(current_balance=p_0, term_length=term,
                                                              note_rate=r)
        balance_array[-1] += self.deferred_balance
        return c, balance_array, interest_array

    def payment_breakdown_generator(self):
        first_pay_date = (self.start_date + relativedelta.relativedelta(months=1)).replace(day=1)
        payment_dates_array = np.zeros(self.term).astype(datetime)
        payment_dates_array[0] = first_pay_date
        for i in range(1, self.term):
            payment_dates_array[i] = (payment_dates_array[i - 1] + relativedelta.relativedelta(months=1)).replace(day=1)
        c, principal, interest = self.mtm_payments_generator()
        df_payments = pd.DataFrame({'Date': payment_dates_array, 'Principal': principal, 'Interest': interest,
                                    'Payment': np.array([c]*self.term)})
        return df_payments

    pass


def frm_term_scheduler(current_balance: float, term_length: int, note_rate: float):
    p_0 = current_balance
    r = note_rate/12
    c = p_0 * r/(1 - (1 + r)**(-term_length))
    balance_array, interest_array = np.empty(term_length), np.empty(term_length)
    balance_array[0] = p_0*(1+r) - c
    interest_array[0] = p_0*r
    for i in range(1, term_length):
        balance_array[i] = balance_array[i - 1] * (1 + r) - c
        interest_array[i] = balance_array[i - 1] * r
    return c, balance_array, interest_array


def io_term_scheduler(current_balance: float, io_term_length: int, effective_rate: float):
    rate_vec = effective_rate[:io_term_length]
    payments = current_balance*rate_vec/12
    interest_array = current_balance*rate_vec/12
    balance_array = np.array([current_balance]*io_term_length)
    return payments, balance_array, interest_array


def arm_term_scheduler(current_balance: float, term_length: int, period: int, note_rate: float):
    pi_payment, balance_array, interest_array = frm_term_scheduler(current_balance=current_balance,
                                                                   term_length=term_length, note_rate=note_rate)
    return np.array([pi_payment]*period), balance_array[:period], interest_array[:period]


class ARM(Mortgage):
    def __init__(self, term: int, total_starting_balance: float, start_date: datetime, initial_term: int,
                 initial_rate: float, ref_rate: str, rate_vector: np.array, io_flag: bool, arm_margin: float,
                 arm_period: int, arm_floor: float, initial_cap: float, rate_adj_cap: float, lifetime_cap: float,
                 deferred_balance=0, io_term=0):
        super().__init__(term, total_starting_balance, start_date, deferred_balance)
        self.initial_term = initial_term
        self.initial_rate = initial_rate
        self.ref_rate = ref_rate
        self.rate_vector = rate_vector
        self.io_flag = io_flag
        self.io_term = io_term
        self.arm_margin = arm_margin
        self.arm_period = arm_period
        self.arm_floor = arm_floor
        self.initial_cap = initial_cap
        self.rate_adj_cap = rate_adj_cap
        self.lifetime_cap = lifetime_cap
        self.effective_rate_vector = np.zeros(self.term)
        self.reset_points = []

    def gen_effective_rate_vector(self):
        self.effective_rate_vector[:self.initial_term] = self.initial_rate
        self.effective_rate_vector[self.initial_term] = max(self.arm_floor,
                                        min(self.initial_rate + min(self.initial_cap, self.lifetime_cap),
                                            self.rate_vector[self.initial_term ] + self.arm_margin))
        for i in range(self.initial_term + 1, self.term):
            if i % self.arm_period == 0:
                self.effective_rate_vector[i] = max(self.arm_floor,
                                        min(self.effective_rate_vector[i - 1] + self.rate_adj_cap,
                                        self.rate_vector[i] + self.arm_margin, self.initial_rate + self.lifetime_cap))
            else:
                self.effective_rate_vector[i] = self.effective_rate_vector[i - 1]

    def gen_total_arm_schedule(self):
        self.gen_effective_rate_vector()
        pi_payments, balance_array, interest_array = np.zeros(self.term), np.zeros(self.term), np.zeros(self.term)
        if self.io_flag:
            pi_payments[:self.io_term], balance_array[:self.io_term], interest_array[:self.io_term] =\
                io_term_scheduler(self.total_starting_balance, self.io_term, self.effective_rate_vector)
            reset_balance = balance_array[self.io_term - 1]
            if self.io_term < self.initial_term:
                pi_payments[self.io_term: self.initial_term], balance_array[self.io_term: self.initial_term], \
                interest_array[self.io_term: self.initial_term] = arm_term_scheduler(reset_balance,
                                            self.term - self.initial_term, self.initial_term - self.io_term,
                                            self.effective_rate_vector[self.io_term + 1])
                reset_balance = balance_array[self.initial_term - 1]
        else:
            pi_payments[: self.initial_term], balance_array[: self.initial_term], \
            interest_array[: self.initial_term] = arm_term_scheduler(self.total_starting_balance,
                                                                                 self.term,
                                                                                 self.initial_term,
                                                                                 self.initial_rate)
            reset_balance = balance_array[self.initial_term - 1]

        if self.io_flag and self.io_term > self.initial_term:
            start_amort_period = self.io_term
        else:
            start_amort_period = self.initial_term
        for i in range(start_amort_period, self.term, self.arm_period):
            pi_payments[i: i + self.arm_period], balance_array[i: i + self.arm_period], \
            interest_array[i: i + self.arm_period] = arm_term_scheduler(reset_balance,
                                                                     self.term - i,
                                                                     self.arm_period,
                                                                     self.effective_rate_vector[i])
            reset_balance = balance_array[i + self.arm_period - 1]

        balance_array[-1] += self.deferred_balance
        return pi_payments, balance_array, interest_array

    pass














