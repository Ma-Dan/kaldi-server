"""Server-end for the ASR demo."""
import os
import time
import random
import argparse
import functools
from time import gmtime, strftime
import socketserver
import struct
import wave
import numpy as np
import socket

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
parser.add_argument('--speech_save_dir',
    default='demo_cache',
    type=str,
    help="Directory to save demo audios.")
args = parser.parse_args()


class AsrTCPServer(socketserver.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 speech_save_dir,
                 audio_process_handler,
                 bind_and_activate=True):
        self.speech_save_dir = speech_save_dir
        self.audio_process_handler = audio_process_handler
        socketserver.TCPServer.__init__(
            self, server_address, RequestHandlerClass, bind_and_activate=True)


class AsrRequestHandler(socketserver.BaseRequestHandler):
    """The ASR request handler."""

    def handle(self):
        # receive data through TCP socket
        chunk = self.request.recv(1024)
        target_len = struct.unpack('>i', chunk[:4])[0]
        data = chunk[4:]
        while len(data) < target_len:
            chunk = self.request.recv(1024)
            data += chunk
        # write to file
        filename = self._write_to_file(data)

        print("Received utterance[length=%d] from %s, saved to %s." %
              (len(data), self.client_address[0], filename))
        start_time = time.time()
        transcript = self.server.audio_process_handler(filename)
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))
        self.request.sendall(transcript.encode('utf-8'))

    def _write_to_file(self, data):
        # prepare save dir and filename
        if not os.path.exists(self.server.speech_save_dir):
            os.mkdir(self.server.speech_save_dir)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        out_filename = os.path.join(
            self.server.speech_save_dir,
            timestamp + "_" + self.client_address[0] + ".wav")
        # write to wav file
        file = wave.open(out_filename, 'wb')
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(data)
        file.close()
        return out_filename

#创建套接字
tcpClientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket---%s'%tcpClientSocket)
#连接服务器
serverAddr = ('127.0.0.1', 6666)
tcpClientSocket.connect(serverAddr)
print('connect success!')

def start_server():
    """Start the ASR server"""

    # prepare ASR inference handler
    def file_to_transcript(filename):
        filename = filename[11:-4]
        tcpClientSocket.send(filename.encode())
        recvData = tcpClientSocket.recv(1024)
        return recvData.decode('utf-8')

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler,
        speech_save_dir=args.speech_save_dir,
        audio_process_handler=file_to_transcript)
    print("ASR Server Started.")
    server.serve_forever()


def main():
    start_server()


if __name__ == "__main__":
    main()
