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

#ifndef PROBING_CLIENT_H
#define PROBING_CLIENT_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-value.h"
#include "ns3/traced-callback.h"

namespace ns3
{

struct ProbingPayload
{
  uint32_t burst;
  Time timestamp;
};

// Typedef for custom callback (TODO: not sure if this is actually needed...)
typedef void (*TracedProbeCallback)(uint32_t burst, Time timestamp);

class Socket;
class Packet;

/**
 * \ingroup probing
 * \brief An active probing client, that is sending probes.
 *
 * Every packet sent should be returned by the server and received here.
 */
class ProbingClient : public Application
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId(void);

  ProbingClient();

  virtual ~ProbingClient();

  /**
   * \brief set the remote address and port
   * \param ip remote IP address
   * \param port remote port
   */
  void SetRemote(Address ip, uint16_t port);
  /**
   * \brief set the remote address
   * \param addr remote address
   */
  void SetRemote(Address addr);

protected:
  virtual void DoDispose(void);

private:
  virtual void StartApplication(void);
  virtual void StopApplication(void);

  /**
   * \brief Schedule the next packet transmission
   * \param dt time interval between packets.
   */
  void ScheduleTransmit(Time dt);
  /**
   * \brief Send a packet
   */
  void Send(void);

  /**
   * \brief Handle a packet reception.
   *
   * This function is called by lower layers.
   *
   * \param socket the socket the packet was received to.
   */
  void HandleRead(Ptr<Socket> socket);

  Time m_interval;      //!< Packet inter-send time
  uint32_t m_burst;     // Counter of packet bursts
  uint32_t m_burstsize; // Packets per burst

  uint32_t m_dataSize; //!< packet payload size (must be equal to m_size)
  uint8_t *m_data;     //!< packet payload data

  uint32_t m_sent;       //!< Counter for sent packets
  Ptr<Socket> m_socket;  //!< Socket
  Address m_peerAddress; //!< Remote peer address
  uint16_t m_peerPort;   //!< Remote peer port
  EventId m_sendEvent;   //!< Event to send the next packet

  //!< Callbacks for measured RTTs
  TracedCallback<uint32_t, Time> m_rttTrace;

  /// Callbacks for tracing the packet Tx events
  TracedCallback<Ptr<const Packet>> m_txTrace;

  /// Callbacks for tracing the packet Rx events
  TracedCallback<Ptr<const Packet>> m_rxTrace;

  /// Callbacks for tracing the packet Tx events, includes source and destination addresses
  TracedCallback<Ptr<const Packet>, const Address &, const Address &> m_txTraceWithAddresses;

  /// Callbacks for tracing the packet Rx events, includes source and destination addresses
  TracedCallback<Ptr<const Packet>, const Address &, const Address &> m_rxTraceWithAddresses;
};

} // namespace ns3

#endif /* PROBING_CLIENT_H */
