/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
#include "ns3/log.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"

#include "probing-client.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ProbingClientApplication");

NS_OBJECT_ENSURE_REGISTERED(ProbingClient);

TypeId
ProbingClient::GetTypeId(void)
{
  static TypeId tid =
      TypeId("ns3::ProbingClient")
          .SetParent<Application>()
          .SetGroupName("Applications")
          .AddConstructor<ProbingClient>()
          .AddAttribute("Interval",
                        "The time to wait between packets",
                        TimeValue(Seconds(1.0)),
                        MakeTimeAccessor(&ProbingClient::m_interval),
                        MakeTimeChecker())
          .AddAttribute("Burstsize",
                        "Number of packets per probe.",
                        UintegerValue(1),
                        MakeUintegerAccessor(&ProbingClient::m_burstsize),
                        MakeUintegerChecker<uint32_t>())
          .AddAttribute("RemoteAddress",
                        "The destination Address of the outbound packets",
                        AddressValue(),
                        MakeAddressAccessor(&ProbingClient::m_peerAddress),
                        MakeAddressChecker())
          .AddAttribute("RemotePort",
                        "The destination port of the outbound packets",
                        UintegerValue(0),
                        MakeUintegerAccessor(&ProbingClient::m_peerPort),
                        MakeUintegerChecker<uint16_t>())
          .AddTraceSource("Tx", "A new packet is created and is sent",
                          MakeTraceSourceAccessor(&ProbingClient::m_txTrace),
                          "ns3::Packet::TracedCallback")
          .AddTraceSource("Rx", "A packet has been received",
                          MakeTraceSourceAccessor(&ProbingClient::m_rxTrace),
                          "ns3::Packet::TracedCallback")
          .AddTraceSource("TxWithAddresses", "A new packet is created and is sent",
                          MakeTraceSourceAccessor(&ProbingClient::m_txTraceWithAddresses),
                          "ns3::Packet::TwoAddressTracedCallback")
          .AddTraceSource("RxWithAddresses", "A packet has been received",
                          MakeTraceSourceAccessor(&ProbingClient::m_rxTraceWithAddresses),
                          "ns3::Packet::TwoAddressTracedCallback")
          .AddTraceSource("RTT", "Measured RTT.",
                          MakeTraceSourceAccessor(&ProbingClient::m_rttTrace),
                          "ns3::ProbingClient::TracedProbeCallback");
  return tid;
}

ProbingClient::ProbingClient()
{
  NS_LOG_FUNCTION(this);
  m_sent = 0;
  m_socket = 0;
  m_sendEvent = EventId();
  m_data = 0;
  m_dataSize = 0;
  m_burst = 0;
}

ProbingClient::~ProbingClient()
{
  NS_LOG_FUNCTION(this);
  m_socket = 0;

  delete[] m_data;
  m_data = 0;
  m_dataSize = 0;
}

void ProbingClient::SetRemote(Address ip, uint16_t port)
{
  NS_LOG_FUNCTION(this << ip << port);
  m_peerAddress = ip;
  m_peerPort = port;
}

void ProbingClient::SetRemote(Address addr)
{
  NS_LOG_FUNCTION(this << addr);
  m_peerAddress = addr;
}

void ProbingClient::DoDispose(void)
{
  NS_LOG_FUNCTION(this);
  Application::DoDispose();
}

void ProbingClient::StartApplication(void)
{
  NS_LOG_FUNCTION(this);

  if (m_socket == 0)
  {
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    m_socket = Socket::CreateSocket(GetNode(), tid);
    if (Ipv4Address::IsMatchingType(m_peerAddress) == true)
    {
      if (m_socket->Bind() == -1)
      {
        NS_FATAL_ERROR("Failed to bind socket");
      }
      m_socket->Connect(InetSocketAddress(Ipv4Address::ConvertFrom(m_peerAddress), m_peerPort));
    }
    else if (Ipv6Address::IsMatchingType(m_peerAddress) == true)
    {
      if (m_socket->Bind6() == -1)
      {
        NS_FATAL_ERROR("Failed to bind socket");
      }
      m_socket->Connect(Inet6SocketAddress(Ipv6Address::ConvertFrom(m_peerAddress), m_peerPort));
    }
    else if (InetSocketAddress::IsMatchingType(m_peerAddress) == true)
    {
      if (m_socket->Bind() == -1)
      {
        NS_FATAL_ERROR("Failed to bind socket");
      }
      m_socket->Connect(m_peerAddress);
    }
    else if (Inet6SocketAddress::IsMatchingType(m_peerAddress) == true)
    {
      if (m_socket->Bind6() == -1)
      {
        NS_FATAL_ERROR("Failed to bind socket");
      }
      m_socket->Connect(m_peerAddress);
    }
    else
    {
      NS_ASSERT_MSG(false, "Incompatible address type: " << m_peerAddress);
    }
  }

  m_socket->SetRecvCallback(MakeCallback(&ProbingClient::HandleRead, this));
  m_socket->SetAllowBroadcast(true);
  ScheduleTransmit(Seconds(0.));
}

void ProbingClient::StopApplication()
{
  NS_LOG_FUNCTION(this);

  if (m_socket != 0)
  {
    m_socket->Close();
    m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    m_socket = 0;
  }

  Simulator::Cancel(m_sendEvent);
}

void ProbingClient::ScheduleTransmit(Time dt)
{
  NS_LOG_FUNCTION(this << dt);
  m_sendEvent = Simulator::Schedule(dt, &ProbingClient::Send, this);
}

void ProbingClient::Send(void)
{
  NS_LOG_FUNCTION(this);

  NS_ASSERT(m_sendEvent.IsExpired());

  Address localAddress;
  m_socket->GetSockName(localAddress);

  // Create packet with current time as payload, to compute diff when it returns
  auto time = Simulator::Now();
  NS_LOG_DEBUG("Sending burst: " << m_burst << " at time: " << time);
  for (uint32_t i = 0; i < m_burstsize; ++i)
  {
    NS_LOG_DEBUG("Sending packet: " << i);
    ProbingPayload data = {m_burst, time};
    auto p = Create<Packet>(reinterpret_cast<uint8_t *>(&data), sizeof(data));

    // call to the trace sinks before the packet is actually sent,
    // so that tags added to the packet can be sent as well
    m_txTrace(p);
    if (Ipv4Address::IsMatchingType(m_peerAddress))
    {
      m_txTraceWithAddresses(p, localAddress, InetSocketAddress(Ipv4Address::ConvertFrom(m_peerAddress), m_peerPort));
    }
    else if (Ipv6Address::IsMatchingType(m_peerAddress))
    {
      m_txTraceWithAddresses(p, localAddress, Inet6SocketAddress(Ipv6Address::ConvertFrom(m_peerAddress), m_peerPort));
    }
    m_socket->Send(p);
    ++m_sent;
  }
  ++m_burst;
  ScheduleTransmit(m_interval);
}

void ProbingClient::HandleRead(Ptr<Socket> socket)
{
  NS_LOG_FUNCTION(this << socket);
  Ptr<Packet> packet;

  Address from;
  Address localAddress;
  while ((packet = socket->RecvFrom(from)))
  {
    if (InetSocketAddress::IsMatchingType(from))
    {
      NS_LOG_INFO("At time " << Simulator::Now().GetSeconds() << "s client received " << packet->GetSize() << " bytes from " << InetSocketAddress::ConvertFrom(from).GetIpv4() << " port " << InetSocketAddress::ConvertFrom(from).GetPort());
    }
    else if (Inet6SocketAddress::IsMatchingType(from))
    {
      NS_LOG_INFO("At time " << Simulator::Now().GetSeconds() << "s client received " << packet->GetSize() << " bytes from " << Inet6SocketAddress::ConvertFrom(from).GetIpv6() << " port " << Inet6SocketAddress::ConvertFrom(from).GetPort());
    }
    socket->GetSockName(localAddress);
    m_rxTrace(packet);
    m_rxTraceWithAddresses(packet, from, localAddress);

    auto current_time = Simulator::Now();
    ProbingPayload data;
    packet->CopyData(reinterpret_cast<uint8_t *>(&data),
                     sizeof(data));
    NS_LOG_DEBUG("Received timestamp: " << data.timestamp);
    NS_LOG_DEBUG("Current time:       " << current_time);
    m_rttTrace(data.burst, current_time - data.timestamp);
  }
}

} // Namespace ns3

int main()
{
  return 0;
}
