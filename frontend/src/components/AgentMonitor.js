import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Card,
    CardHeader,
    CardBody,
    Heading,
    Text,
    VStack,
    HStack,
    Badge,
    Progress,
    useColorModeValue,
    Accordion,
    AccordionItem,
    AccordionButton,
    AccordionPanel,
    AccordionIcon,
    Code,
    Divider,
    Button,
    Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    SimpleGrid,
    Icon,
} from '@chakra-ui/react';
import {
    FiCompass,
    FiArchive,
    FiRefreshCw,
    FiCheckCircle,
    FiZap,
    FiShuffle,
    FiSun,
    FiThermometer,
    FiSliders,
    FiTarget,
    FiCpu,
    FiLink,
    FiStar,
    FiTrendingUp,
    FiEdit3,
    FiAlertCircle,
    FiTool,
    FiGrid,
} from 'react-icons/fi';

const AgentMonitor = ({ jobId, onComplete }) => {
    const [events, setEvents] = useState([]);
    const [agents, setAgents] = useState({});
    const [summary, setSummary] = useState({
        total_events: 0,
        agents_active: 0,
        agents_complete: 0,
        examples_generated: 0,
    });
    const eventSourceRef = useRef(null);
    const eventsEndRef = useRef(null);

    const cardBg = useColorModeValue('white', 'gray.700');
    const borderColor = useColorModeValue('gray.200', 'gray.600');
    const eventBg = useColorModeValue('gray.50', 'gray.800');

    useEffect(() => {
        if (!jobId) return;

        // Connect to SSE stream
        const eventSource = new EventSource(`http://localhost:8000/api/generator/stream/${jobId}`);
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'connected' || data.type === 'keepalive') {
                    return;
                }

                // Add event to history
                setEvents(prev => [...prev, data]);

                // Update agent state
                if (data.agent_id) {
                    setAgents(prev => ({
                        ...prev,
                        [data.agent_id]: {
                            ...prev[data.agent_id],
                            role: data.agent_role,
                            last_event: data.event_type,
                            last_update: data.timestamp,
                            data: data.data,
                        }
                    }));
                }

                // Update summary
                setSummary(prev => ({
                    ...prev,
                    total_events: prev.total_events + 1,
                    agents_active: Object.values(agents).filter(a => a.last_event !== 'complete').length,
                    agents_complete: Object.values(agents).filter(a => a.last_event === 'complete').length,
                }));

                // Check if generation is complete
                if (data.event_type === 'complete' && data.agent_role === 'selector') {
                    if (onComplete) {
                        onComplete(data);
                    }
                }
            } catch (error) {
                console.error('Error parsing SSE event:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            eventSource.close();
        };

        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, [jobId]);

    // Auto-scroll to latest event
    useEffect(() => {
        eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [events]);

    const roleIconMap = {
        // Swarm roles
        scout: FiCompass,
        gatherer: FiArchive,
        mutator: FiRefreshCw,
        selector: FiCheckCircle,
        // Evolution roles
        mutagen: FiZap,
        crossover: FiShuffle,
        // Cosmic roles
        exploder: FiSun,
        cooler: FiThermometer,
        synthesizer: FiSliders,
        // Quantum roles
        gauge: FiTarget,
        fermion: FiCpu,
        yukawa: FiLink,
        higgs: FiStar,
        potential: FiTrendingUp,
        // Adversarial multi-model roles
        generator: FiEdit3,
        critic: FiAlertCircle,
        refiner: FiTool,
        diversity_checker: FiGrid,
    };

    const getAgentIcon = (role) => roleIconMap[role] || FiCpu;

    const getEventColor = (eventType) => {
        const colors = {
            start: 'blue',
            thinking: 'purple',
            streaming: 'cyan',
            output: 'green',
            complete: 'teal',
            error: 'red',
            metadata: 'gray',
        };
        return colors[eventType] || 'gray';
    };

    return (
        <Box>
            {/* Summary Stats */}
            <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4} mb={6}>
                <Stat>
                    <StatLabel>Total Events</StatLabel>
                    <StatNumber>{summary.total_events}</StatNumber>
                </Stat>
                <Stat>
                    <StatLabel>Active Agents</StatLabel>
                    <StatNumber>{summary.agents_active}</StatNumber>
                </Stat>
                <Stat>
                    <StatLabel>Completed</StatLabel>
                    <StatNumber>{summary.agents_complete}</StatNumber>
                </Stat>
                <Stat>
                    <StatLabel>Examples</StatLabel>
                    <StatNumber>{summary.examples_generated}</StatNumber>
                </Stat>
            </SimpleGrid>

            {/* Agent Cards */}
            <Heading size="sm" mb={4}>Active Agents</Heading>
            <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4} mb={6}>
                {Object.entries(agents).map(([agentId, agent]) => (
                    <Card key={agentId} bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                        <CardHeader pb={2}>
                            <HStack justify="space-between">
                                <HStack>
                                    <Icon as={getAgentIcon(agent.role)} boxSize={6} color="gray.500" />
                                    <VStack align="start" spacing={0}>
                                        <Text fontWeight="bold" fontSize="sm">{agentId}</Text>
                                        <Text fontSize="xs" color="gray.500">{agent.role}</Text>
                                    </VStack>
                                </HStack>
                                <Badge colorScheme={getEventColor(agent.last_event)}>
                                    {agent.last_event}
                                </Badge>
                            </HStack>
                        </CardHeader>
                        <CardBody pt={2}>
                            {agent.data?.thought && (
                                <Text fontSize="sm" color="gray.600" fontStyle="italic">
                                    💭 "{agent.data.thought}"
                                </Text>
                            )}
                            {agent.data?.message && (
                                <Text fontSize="sm" color="gray.600">
                                    {agent.data.message}
                                </Text>
                            )}
                        </CardBody>
                    </Card>
                ))}
            </SimpleGrid>

            {/* Event Timeline */}
            <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                <CardHeader>
                    <HStack justify="space-between">
                        <Heading size="sm">Event Timeline</Heading>
                        <Button size="sm" onClick={() => setEvents([])}>Clear</Button>
                    </HStack>
                </CardHeader>
                <CardBody>
                    <Box maxH="400px" overflowY="auto">
                        <VStack align="stretch" spacing={2}>
                            {events.map((event, index) => (
                                <Box
                                    key={index}
                                    p={3}
                                    bg={eventBg}
                                    borderRadius="md"
                                    borderLeftWidth="3px"
                                    borderLeftColor={`${getEventColor(event.event_type)}.500`}
                                >
                                    <HStack justify="space-between" mb={1}>
                                        <HStack>
                                            <Icon as={getAgentIcon(event.agent_role)} boxSize={4} color="gray.500" />
                                            <Text fontSize="sm" fontWeight="bold">
                                                {event.agent_id}
                                            </Text>
                                            <Badge colorScheme={getEventColor(event.event_type)} size="sm">
                                                {event.event_type}
                                            </Badge>
                                        </HStack>
                                        <Text fontSize="xs" color="gray.500">
                                            {new Date(event.timestamp * 1000).toLocaleTimeString()}
                                        </Text>
                                    </HStack>

                                    {event.data?.thought && (
                                        <Text fontSize="sm" fontStyle="italic" color="purple.600">
                                            💭 {event.data.thought}
                                        </Text>
                                    )}

                                    {event.data?.partial_output && (
                                        <Code fontSize="xs" p={2} borderRadius="md" display="block" whiteSpace="pre-wrap">
                                            {event.data.partial_output.substring(0, 200)}
                                            {event.data.partial_output.length > 200 && '...'}
                                        </Code>
                                    )}

                                    {event.data?.message && (
                                        <Text fontSize="sm">{event.data.message}</Text>
                                    )}
                                </Box>
                            ))}
                            <div ref={eventsEndRef} />
                        </VStack>
                    </Box>
                </CardBody>
            </Card>
        </Box>
    );
};

export default AgentMonitor;
