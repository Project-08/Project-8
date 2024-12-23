import pyamgx

# pyamgx must be initialized globally - the safest bet is to just do it here
pyamgx.initialize()
pyamgx.install_signal_handler()
