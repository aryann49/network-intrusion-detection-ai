from scapy.all import sniff

def packet_callback(packet):
    print(packet.summary())

print("Starting packet capture...")

sniff(
    iface="Wi-Fi",
    prn=packet_callback,
    store=False
)