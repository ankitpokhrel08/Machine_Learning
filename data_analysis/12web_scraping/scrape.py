from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time



s = Service('/opt/homebrew/bin/chromedriver')
driver = webdriver.Chrome(service=s)
driver.get("https://kalimatimarket.gov.np/price")
time.sleep(2)

input("Click...")