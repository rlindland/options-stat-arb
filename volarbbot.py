from tradersbot import *

from time import sleep
from time import time

import blackscholes
import numpy as np
from scipy.stats import norm
import math
import copy

t = TradersBot('127.0.0.1', 'trader0', 'trader0')

DEFAULT_POSITION_LIMIT = 5000

class Options():
    def __init__(self, default=None):
        self.data = default or {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, val):
        if key not in self.data:
            print("Invalid option: %s" % key)
        else:
            print("Set %s to %s" % (key, val))
            self.data[key] = val

class BaseBot():
    def __init__(self):
        self.options = Options({
            'delay': 1,
            'position_limit': DEFAULT_POSITION_LIMIT,
            'order_quantity': 10,
        })
        self.count = 0
        self.deltas = 0
        self.vegas = 0
        self.elapsedTime = 0
        self.topBid = {}
        self.topAsk = {}
        self.lastPrices = {"TMXFUT": 100,
                           'T80P': -1,
                           'T80C': -1,
                           'T81P': -1,
                           'T81C': -1,
                           'T82P': -1,
                           'T82C': -1,
                           'T83P': -1,
                           'T83C': -1,
                           'T84P': -1,
                           'T84C': -1,
                           'T85P': -1,
                           'T85C': -1,
                           'T86P': -1,
                           'T86C': -1,
                           'T87P': -1,
                           'T87C': -1,
                           'T88P': -1,
                           'T88C': -1,
                           'T89P': -1,
                           'T89C': -1,
                           'T90P': -1,
                           'T90C': -1,
                           'T91P': -1,
                           'T91C': -1,
                           'T92P': -1,
                           'T92C': -1,
                           'T93P': -1,
                           'T93C': -1,
                           'T94P': -1,
                           'T94C': -1,
                           'T95P': -1,
                           'T95C': -1,
                           'T96P': -1,
                           'T96C': -1,
                           'T97P': -1,
                           'T97C': -1,
                           'T98P': -1,
                           'T98C': -1,
                           'T99P': -1,
                           'T99C': -1,
                           'T100P': -1,
                           'T100C': -1,
                           'T101P': -1,
                           'T101C': -1,
                           'T102P': -1,
                           'T102C': -1,
                           'T103P': -1,
                           'T103C': -1,
                           'T104P': -1,
                           'T104C': -1,
                           'T105P': -1,
                           'T105C': -1,
                           'T106P': -1,
                           'T106C': -1,
                           'T107P': -1,
                           'T107C': -1,
                           'T108P': -1,
                           'T108C': -1,
                           'T109P': -1,
                           'T109C': -1,
                           'T110P': -1,
                           'T110C': -1,
                           'T111P': -1,
                           'T111C': -1,
                           'T112P': -1,
                           'T112C': -1,
                           'T113P': -1,
                           'T113C': -1,
                           'T114P': -1,
                           'T114C': -1,
                           'T115P': -1,
                           'T115C': -1,
                           'T116P': -1,
                           'T116C': -1,
                           'T117P': -1,
                           'T117C': -1,
                           'T118P': -1,
                           'T118C': -1,
                           'T119P': -1,
                           'T119C': -1,
                           'T120P': -1,
                           'T120C': -1, }
        self.positions = {'T80P': 0,
                          'T80C': 0,
                          'T81P': 0,
                          'T81C': 0,
                          'T82P': 0,
                          'T82C': 0,
                          'T83P': 0,
                          'T83C': 0,
                          'T84P': 0,
                          'T84C': 0,
                          'T85P': 0,
                          'T85C': 0,
                          'T86P': 0,
                          'T86C': 0,
                          'T87P': 0,
                          'T87C': 0,
                          'T88P': 0,
                          'T88C': 0,
                          'T89P': 0,
                          'T89C': 0,
                          'T90P': 0,
                          'T90C': 0,
                          'T91P': 0,
                          'T91C': 0,
                          'T92P': 0,
                          'T92C': 0,
                          'T93P': 0,
                          'T93C': 0,
                          'T94P': 0,
                          'T94C': 0,
                          'T95P': 0,
                          'T95C': 0,
                          'T96P': 0,
                          'T96C': 0,
                          'T97P': 0,
                          'T97C': 0,
                          'T98P': 0,
                          'T98C': 0,
                          'T99P': 0,
                          'T99C': 0,
                          'T100P': 0,
                          'T100C': 0,
                          'T101P': 0,
                          'T101C': 0,
                          'T102P': 0,
                          'T102C': 0,
                          'T103P': 0,
                          'T103C': 0,
                          'T104P': 0,
                          'T104C': 0,
                          'T105P': 0,
                          'T105C': 0,
                          'T106P': 0,
                          'T106C': 0,
                          'T107P': 0,
                          'T107C': 0,
                          'T108P': 0,
                          'T108C': 0,
                          'T109P': 0,
                          'T109C': 0,
                          'T110P': 0,
                          'T110C': 0,
                          'T111P': 0,
                          'T111C': 0,
                          'T112P': 0,
                          'T112C': 0,
                          'T113P': 0,
                          'T113C': 0,
                          'T114P': 0,
                          'T114C': 0,
                          'T115P': 0,
                          'T115C': 0,
                          'T116P': 0,
                          'T116C': 0,
                          'T117P': 0,
                          'T117C': 0,
                          'T118P': 0,
                          'T118C': 0,
                          'T119P': 0,
                          'T119C': 0,
                          'T120P': 0,
                          'T120C': 0,
                          'TMXFUT': 0, }
        self.expect_positions = {'T80P': 0,
                                 'T80C': 0,
                                 'T81P': 0,
                                 'T81C': 0,
                                 'T82P': 0,
                                 'T82C': 0,
                                 'T83P': 0,
                                 'T83C': 0,
                                 'T84P': 0,
                                 'T84C': 0,
                                 'T85P': 0,
                                 'T85C': 0,
                                 'T86P': 0,
                                 'T86C': 0,
                                 'T87P': 0,
                                 'T87C': 0,
                                 'T88P': 0,
                                 'T88C': 0,
                                 'T89P': 0,
                                 'T89C': 0,
                                 'T90P': 0,
                                 'T90C': 0,
                                 'T91P': 0,
                                 'T91C': 0,
                                 'T92P': 0,
                                 'T92C': 0,
                                 'T93P': 0,
                                 'T93C': 0,
                                 'T94P': 0,
                                 'T94C': 0,
                                 'T95P': 0,
                                 'T95C': 0,
                                 'T96P': 0,
                                 'T96C': 0,
                                 'T97P': 0,
                                 'T97C': 0,
                                 'T98P': 0,
                                 'T98C': 0,
                                 'T99P': 0,
                                 'T99C': 0,
                                 'T100P': 0,
                                 'T100C': 0,
                                 'T101P': 0,
                                 'T101C': 0,
                                 'T102P': 0,
                                 'T102C': 0,
                                 'T103P': 0,
                                 'T103C': 0,
                                 'T104P': 0,
                                 'T104C': 0,
                                 'T105P': 0,
                                 'T105C': 0,
                                 'T106P': 0,
                                 'T106C': 0,
                                 'T107P': 0,
                                 'T107C': 0,
                                 'T108P': 0,
                                 'T108C': 0,
                                 'T109P': 0,
                                 'T109C': 0,
                                 'T110P': 0,
                                 'T110C': 0,
                                 'T111P': 0,
                                 'T111C': 0,
                                 'T112P': 0,
                                 'T112C': 0,
                                 'T113P': 0,
                                 'T113C': 0,
                                 'T114P': 0,
                                 'T114C': 0,
                                 'T115P': 0,
                                 'T115C': 0,
                                 'T116P': 0,
                                 'T116C': 0,
                                 'T117P': 0,
                                 'T117C': 0,
                                 'T118P': 0,
                                 'T118C': 0,
                                 'T119P': 0,
                                 'T119C': 0,
                                 'T120P': 0,
                                 'T120C': 0,
                                 'TMXFUT': 0, }
        self.maxPos = 20000
        self.priceChange = {}
        self.pnl = 0
        self.avg_vol = {}
        for i in range(80,121):
            x = float(i)
            self.avg_vol[x] = 0.15
        #self.avg_vol = {90.: 0.15, 95.: 0.15, 100.: 0.15, 105.: 0.15, 110.: 0.15}

        self.iv_diff_threshold = 0.0001

    def value(self):
        val = 0
        for key in self.positions:
            val += self.positions[key] * self.lastPrices[key]
        return val

    def volatility_smile(self):
        prices = []
        for i in range(41):
            prices.append(80.+i)
        ivs = []
        deltas = []
        vegas = []
        calls = []
        puts = []
        if 'TMXFUT' in self.lastPrices:
            for p in prices:
                s = self.lastPrices['TMXFUT']
                k = p
                r = 0
                t = 0
                big_T = (1. / 12.) * (900. - self.elapsedTime) / 900.
                #What is 12?
                call = self.lastPrices['T' + str(int(p)) + 'C']
                put = self.lastPrices['T' + str(int(p)) + 'P']
                eps = .000001
                o = blackscholes.invert_scalar(s, k, r, t, big_T, call, put, eps)
                ivs.append(o)
                d1 = (math.log(s / k) + (r + o * o / 2) * (big_T - t)) / (o * math.sqrt(big_T - t))
                deltas.append(norm.cdf(d1) * 100.)
                vegas.append(s * norm.pdf(d1) * math.sqrt(big_T - t))
                calls.append(call)
                puts.append(put)
            return prices, ivs, deltas, vegas, calls, puts
        pass

    def delta(self):
        delta = 0
        prices = []
        for i in range(41):
            prices.append(80.+i)
        deltas = self.volatility_smile()[2]
        for p in prices:
            call_name = 'T' + str(int(p)) + 'C'
            put_name = 'T' + str(int(p)) + 'P'
            delta -= self.positions[put_name] * deltas[prices.index(p)]
            delta += self.positions[call_name] * deltas[prices.index(p)]
        delta += self.positions["TMXFUT"] * 100
        return delta

    def arbitrage(self, msg, order):

        prices = []
        for i in range(41):
            prices.append(80.+i)

        for p in prices:
            k = p
            call_name = 'T' + str(int(p)) + 'C'
            put_name = 'T' + str(int(p)) + 'P'
            q = self.options.get("order_quantity")

            if (
                                    call_name in self.topBid and call_name in self.topAsk and put_name in self.topBid and put_name in self.topAsk and 'TMXFUT' in self.topBid and 'TMXFUT' in self.topAsk):
                big_T = (1. / 12.) * (900. - self.elapsedTime) / 900.
                call = self.lastPrices[call_name]
                put = self.lastPrices[put_name]
                iv = blackscholes.invert_scalar(self.lastPrices['TMXFUT'], k, 0, 0, big_T, call, put, .00001)

                diff = abs(iv - self.avg_vol[k])

                if (iv < self.avg_vol[k] - self.iv_diff_threshold):
                    if (self.positions[call_name] < self.maxPos * diff):
                        order.addBuy(call_name, q, self.topAsk[call_name] + 5)
                        print("YES")
                    if (self.positions[put_name] < self.maxPos * diff):
                        order.addBuy(put_name, q, self.topAsk[put_name] + 5)
                        print("YES")

                elif (iv > self.avg_vol[k] + self.iv_diff_threshold * diff):
                    if (self.positions[call_name] > -self.maxPos):
                        order.addSell(call_name, q, max(0, self.topBid[call_name] - 5))
                        print("YES")
                    if (self.positions[put_name] > -self.maxPos * diff):
                        order.addSell(put_name, q, max(0, self.topBid[put_name] - 5))
                        print("YES")

    def arbitrage2(self,msg,order):

        mindiscrep = 30

        prices = []
        for i in range(41):
            prices.append(80.+i)

        ivs = self.volatility_smile()[2]

        coefs = self.fitIvs(ivs)

        expected = []
        for i in range(80,121):
            expected.append(coefs[0]*i**4+coefs[1]*i**3+coefs[2]*i*i+coefs[3]*i+coefs[4])

        for j in range(len(prices)):
            call_name = 'T' + str(int(j+80)) + 'C'
            put_name = 'T' + str(int(j+80)) + 'P'
            if prices[j] - expected[j] > mindiscrep:
                print("buy")
                if call_name in self.topAsk:
                    order.addBuy(call_name, 1, self.topAsk[call_name] + 5)
                if put_name in self.topBid:
                    order.addSell(put_name, 1, self.topBid[put_name] - 5)
            elif expected[j] - prices[j] > mindiscrep:
                print("sell")
                if call_name in self.topBid:
                    order.addSell(call_name, 1, max(0, self.topBid[call_name] - 5))
                if put_name in self.topAsk:
                    order.addBuy(put_name, 1, self.topAsk[put_name] + 5)

    def hedge_the_fuck(self, msg, order):
        if self.delta() is not None:
            print("N")
            n = min(500, math.floor(abs(self.delta()) / 100))
            print(n)
            if n < 0:
                order.addBuy("TMXFUT", n, self.topAsk["TMXFUT"] + 5)
                print("YES")
            if n > 0:
                order.addSell("TMXFUT", n, max(0, self.topBid["TMXFUT"] - 5))
                print("YES")
            print("Order made.")
        else:
            print("No such orders.")

    def update(self, msg):
        if type(msg.get('elapsed_time')) == int:
            self.elapsedTime = msg.get('elapsed_time')

        # Update internal positions

        if msg.get('market_states'):
            for ticker, state in msg['market_states'].iteritems():
                #print(ticker)
                if len(state['bids']):
                    self.topBid[ticker] = max(map(float, state['bids'].keys()))
                if len(state['asks']):
                    self.topAsk[ticker] = min(map(float, state['asks'].keys()))
                self.lastPrices[ticker] = state['last_price']
                self.priceChange[ticker] = 0

        # Update internal book for a single ticker
        if msg.get('market_state'):
            state = msg['market_state']
            ticker = state['ticker']
            #print(ticker)

            if len(state['bids']):
                self.topBid[ticker] = max(map(float, state['bids'].keys()))
            if len(state['asks']):
                self.topAsk[ticker] = min(map(float, state['asks'].keys()))

            self.lastPrices[ticker] = state['last_price']

        else:
            return None

    def update_positions(self, msg, order):
        if msg.get('trader_state'):
            self.positions = msg['trader_state']['positions']
        if msg.get('message_type') == 'TRADER UPDATE':
            self.cash = msg["trader_state"]["cash"]["USD"]
            self.pnl = msg["trader_state"]["pnl"]["USD"]

    def process(self, msg, order):
        allInit = True
        for x in self.lastPrices:
            if x == -1:
                allInit = False

        if allInit:
            print(msg['market_state'])
            if msg.get('market_state') or msg.get('trader_state') or msg.get('market_states'):
                self.update(msg)
                if self.count % 10 == 0:
                    self.arbitrage2(msg, order)
                    #print(self.pnl)
                
               # print(self.positions)
                self.count += 1
            

    def fitIvs(self, ivs):
        x = []
        for i in range(41):
            x.append(80.+i)
        return np.polyfit(x,ivs,4)
    def place_orders(self, order):
        pass


    def g(msg, order):
        pass
        global tokens
        global tick
        global ticks
        global ids
        if 'orders' in msg:
            for trade in msg['orders']:
                print(trade['ticker'],trade['price'])
                order_id = trade['order_id']
                ticker = trade['ticker']
                token = msg['token']
                time = tokens[token]
                if time in ids:
                    ids[time].append((order_id, ticker))
                else:
                    ids[time] = [(order_id,ticker)] 
b = BaseBot()

t.onMarketUpdate = b.process
t.onTraderUpdate = b.update_positions
t.run()