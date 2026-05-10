import os

from scapy.all import sniff


def packet_callback(packet):
    print(packet.summary())


def main():
    iface = os.environ.get("IDS_SNIFF_IFACE", "Wi-Fi")
    print(f"Starting packet capture on interface: {iface}")

    sniff(
        iface=iface,
        prn=packet_callback,
        store=False,
    )


if __name__ == "__main__":
    main()
