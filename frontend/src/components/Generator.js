import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Select,
  Stack,
  Text,
  Textarea,
  useToast,
  Alert,
  AlertIcon,
  Progress,
  Badge,
  Code,
  Flex,
  SimpleGrid,
  VStack,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Switch,
  Divider,
  Radio,
  RadioGroup,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  IconButton,
  Tooltip,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Icon,
  InputGroup,
  InputLeftElement,
  Menu,
  MenuButton,
  MenuList,
  MenuGroup,
  MenuItem,
  MenuDivider,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  ModalFooter,
} from '@chakra-ui/react';
import {
  FiSettings,
  FiHelpCircle,
  FiCompass,
  FiArchive,
  FiSearch,
  FiChevronDown,
  FiFileText,
  FiX,
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
} from 'react-icons/fi';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { getDomains, getProviders, getProviderModels, startGenerator, getGeneratorStatus, getTemplates } from '../services/api';
import DomainCard from './common/DomainCard';
import AgentMonitor from './AgentMonitor';

// Domain labels for template grouping
const DOMAIN_LABELS = {
  general: 'General Assistant',
  support: 'Customer Support',
  medical: 'Medical Information',
  legal: 'Legal Documentation',
  education: 'Educational Content',
  educational: 'Educational Content',
  business: 'Business Communication',
  sales: 'Sales & Negotiation',
  financial: 'Financial Analysis',
  marketing: 'Marketing',
  hr: 'HR & Recruitment',
  ecommerce: 'E-commerce',
  meetings: 'Meeting Summaries',
  coaching: 'Coaching & Mentoring',
  research: 'Research & Data',
  creative: 'Creative Writing',
  gaming: 'Gaming',
  data: 'Data Analysis',
  coding: 'Coding & Technical',
  technical: 'Technical',
};

const Generator = () => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [selectedSubdomain, setSelectedSubdomain] = useState(null);
  const [generationParams, setGenerationParams] = useState({
    format: 'chat',
    language: 'en',
    count: 50,
    temperature: 0.7,
    provider: 'ollama',
    model: '',
    template: '',
    generation_mode: 'standard', // 'standard' or 'advanced'
    advanced_method: 'swarm', // 'swarm', 'evolution', 'cosmic', 'quantum'
  });
  const [agentModels, setAgentModels] = useState({
    scout: { provider: 'ollama', model: '' },
    gatherer: { provider: 'ollama', model: '' },
    mutator: { provider: 'ollama', model: '' },
    selector: { provider: 'ollama', model: '' },
    mutagen: { provider: 'ollama', model: '' },
    crossover: { provider: 'ollama', model: '' },
    exploder: { provider: 'ollama', model: '' },
    cooler: { provider: 'ollama', model: '' },
    synthesizer: { provider: 'ollama', model: '' },
    gauge: { provider: 'ollama', model: '' },
    fermion: { provider: 'ollama', model: '' },
    yukawa: { provider: 'ollama', model: '' },
    higgs: { provider: 'ollama', model: '' },
    potential: { provider: 'ollama', model: '' },
  });
  const [currentStep, setCurrentStep] = useState(0);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [showAgentMonitor, setShowAgentMonitor] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure(); // For template modal
  const { isOpen: isPreviewOpen, onOpen: onPreviewOpen, onClose: onPreviewClose } = useDisclosure(); // For preview modal
  const [customTemplate, setCustomTemplate] = useState('');
  const [templateSearch, setTemplateSearch] = useState('');
  const [previewExample, setPreviewExample] = useState(null);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);

  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const accordionHoverBg = useColorModeValue('gray.50', 'gray.600');

  // Fetch providers
  const { data: providersData } = useQuery({
    queryKey: ['providers'],
    queryFn: getProviders,
  });

  // Fetch models for all providers
  const [providerModels, setProviderModels] = useState({});

  useEffect(() => {
    const fetchModels = async () => {
      if (providersData?.providers && Array.isArray(providersData.providers)) {
        const models = {};

        // Load custom models from localStorage
        const savedCustomModels = localStorage.getItem('custom_models');
        const customModels = savedCustomModels ? JSON.parse(savedCustomModels) : {};

        for (const provider of providersData.providers) {
          try {
            const data = await getProviderModels(provider.id);

            // Merge API models with custom models
            const apiModels = Array.isArray(data) ? data : [];
            const custom = customModels[provider.id] || [];

            // Combine and deduplicate
            const combined = [...apiModels];
            custom.forEach(modelName => {
              if (!combined.find(m => m.id === modelName)) {
                combined.push({ id: modelName, name: modelName });
              }
            });

            models[provider.id] = combined;
            console.log(`Loaded models for ${provider.id}:`, combined);
          } catch (error) {
            console.error(`Failed to fetch models for ${provider.id}:`, error);

            // If API fails, at least show custom models
            const custom = customModels[provider.id] || [];
            models[provider.id] = custom.map(name => ({ id: name, name }));
          }
        }
        setProviderModels(models);
        console.log('All provider models:', models);
      }
    };
    fetchModels();
  }, [providersData]);

  // Fetch domains
  const {
    data: domainsData,
    isLoading: isLoadingDomains,
    isError: isDomainsError,
    error: domainsError,
    refetch: refetchDomains
  } = useQuery({
    queryKey: ['domains'],
    queryFn: getDomains,
    retry: 1
  });

  // Fetch models for selected provider
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models', generationParams.provider],
    queryFn: () => getProviderModels(generationParams.provider),
    enabled: !!generationParams.provider
  });

  // Fetch templates
  const { data: templatesData = [], isLoading: isLoadingTemplates } = useQuery({
    queryKey: ['templates'],
    queryFn: getTemplates,
  });

  // Group templates by domain for easier selection
  const groupedTemplates = React.useMemo(() => {
    if (!templatesData || templatesData.length === 0) return {};

    const groups = {};
    templatesData.forEach(template => {
      const domain = template.domain || 'general';
      if (!groups[domain]) {
        groups[domain] = [];
      }
      groups[domain].push(template);
    });

    // Sort templates within each group by name
    Object.keys(groups).forEach(domain => {
      groups[domain].sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    });

    return groups;
  }, [templatesData]);

  // Filter templates by search query
  const filteredTemplates = React.useMemo(() => {
    if (!templateSearch.trim()) return groupedTemplates;

    const search = templateSearch.toLowerCase();
    const filtered = {};

    Object.entries(groupedTemplates).forEach(([domain, templates]) => {
      const matching = templates.filter(t =>
        (t.name || '').toLowerCase().includes(search) ||
        (t.description || '').toLowerCase().includes(search) ||
        domain.toLowerCase().includes(search)
      );
      if (matching.length > 0) {
        filtered[domain] = matching;
      }
    });

    return filtered;
  }, [groupedTemplates, templateSearch]);

  // Get selected template object
  const selectedTemplate = React.useMemo(() => {
    if (!generationParams.template) return null;
    return templatesData.find(t => t.id === generationParams.template || t.name === generationParams.template);
  }, [templatesData, generationParams.template]);

  // Fetch job status
  useEffect(() => {
    let interval;
    if (jobId) {
      interval = setInterval(async () => {
        try {
          const status = await getGeneratorStatus(jobId);
          setJobStatus(status);

          if (['completed', 'failed', 'cancelled'].includes(status.status)) {
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error fetching job status:', error);
          clearInterval(interval);
        }
      }, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId]);

  useEffect(() => {
    if (jobStatus?.status === 'completed') {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
    }
  }, [jobStatus, queryClient]);

  // Start generation mutation
  const startGenerationMutation = useMutation({
    mutationFn: startGenerator,
    onSuccess: (data) => {
      setJobId(data.job_id);
      setCurrentStep(2);
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
      toast({
        title: 'Generation started',
        description: `Job ID: ${data.job_id}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    },
    onError: (error) => {
      toast({
        title: 'Error starting generation',
        description: error.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const handleDomainSelect = (domain) => {
    setSelectedDomain(domain);
    setSelectedSubdomain(null);
    setCurrentStep(1);
  };

  const handleSubdomainSelect = (subdomain) => {
    setSelectedSubdomain(subdomain);
  };

  const handleParamChange = (key, value) => {
    setGenerationParams({
      ...generationParams,
      [key]: value,
      // Reset model when provider changes
      ...(key === 'provider' ? { model: '' } : {})
    });
  };

  const handleStartGeneration = () => {
    // Validate model if in standard mode
    if (generationParams.generation_mode === 'standard' && !generationParams.model) {
      toast({
        title: 'Model Required',
        description: 'Please select a model to proceed with generation.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    // Validate agent models if in advanced mode
    if (generationParams.generation_mode === 'advanced') {
      const roles = getAgentRolesForMethod(generationParams.advanced_method);
      const missingModels = roles.filter(role => !agentModels[role]?.model);

      if (missingModels.length > 0) {
        toast({
          title: 'Agent Models Required',
          description: `Please select models for all agents: ${missingModels.map(r => r.charAt(0).toUpperCase() + r.slice(1)).join(', ')}`,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        return;
      }
    }

    const params = {
      domain: selectedDomain.key,
      subdomain: selectedSubdomain?.key,
      ...generationParams,
    };

    // Add agent models if in advanced mode
    if (generationParams.generation_mode === 'advanced') {
      params.agent_models = {};
      getAgentRolesForMethod(generationParams.advanced_method).forEach(role => {
        if (agentModels[role]?.model) {
          params.agent_models[role] = {
            provider: agentModels[role].provider,
            model: agentModels[role].model,
          };
        }
      });
    }

    startGenerationMutation.mutate(params);
  };

  const handleReset = () => {
    setSelectedDomain(null);
    setSelectedSubdomain(null);
    setJobId(null);
    setJobStatus(null);
    setCurrentStep(0);
  };

  const handleGeneratePreview = async () => {
    // Validate model if in standard mode
    if (generationParams.generation_mode === 'standard' && !generationParams.model) {
      toast({
        title: 'Model Required',
        description: 'Please select a model to proceed with preview generation.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    // Validate agent models if in advanced mode
    if (generationParams.generation_mode === 'advanced') {
      const roles = getAgentRolesForMethod(generationParams.advanced_method);
      const missingModels = roles.filter(role => !agentModels[role]?.model);

      if (missingModels.length > 0) {
        toast({
          title: 'Agent Models Required',
          description: `Please select models for all agents: ${missingModels.map(r => r.charAt(0).toUpperCase() + r.slice(1)).join(', ')}`,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
        return;
      }
    }

    setIsGeneratingPreview(true);
    try {
      const params = {
        domain: selectedDomain.key,
        subdomain: selectedSubdomain?.key,
        ...generationParams,
        count: 1, // Only generate 1 example for preview
      };

      // Add agent models if in advanced mode
      if (generationParams.generation_mode === 'advanced') {
        params.agent_models = {};
        getAgentRolesForMethod(generationParams.advanced_method).forEach(role => {
          if (agentModels[role]?.model) {
            params.agent_models[role] = {
              provider: agentModels[role].provider,
              model: agentModels[role].model,
            };
          }
        });
      }

      // Call the generation API
      const response = await startGenerator(params);

      // Set job ID and switch to progress view to show logs
      setJobId(response.job_id);
      setCurrentStep(2);
      setIsGeneratingPreview(false);

      toast({
        title: 'Generating preview',
        description: 'Watch the generation progress below',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate preview';
      const suggestion = errorMessage.includes('model')
        ? 'Try selecting a different model or check if your LLM provider is running.'
        : errorMessage.includes('domain')
          ? 'Please select both domain and subdomain before generating.'
          : 'Check your network connection and try again.';

      toast({
        title: 'Preview generation failed',
        description: `${errorMessage}\n\n💡 ${suggestion}`,
        status: 'error',
        duration: 7000,
        isClosable: true,
      });
      setIsGeneratingPreview(false);
    }
  };

  // Calculate generation progress
  const calculateProgress = () => {
    if (!jobStatus) return 0;
    return (jobStatus.examples_generated / jobStatus.examples_requested) * 100;
  };

  // Get consistent color scheme for domain (same as DomainCard)
  const getDomainColorScheme = (domainId) => {
    const colorMap = {
      support: 'blue',
      medical: 'red',
      legal: 'purple',
      education: 'green',
      business: 'cyan',
      technical: 'orange',
      sales: 'pink',
      financial: 'teal',
      research: 'purple',
      coaching: 'yellow',
      creative: 'pink',
      meetings: 'gray',
      ecommerce: 'blue',
      hr: 'green',
      marketing: 'pink',
      gaming: 'purple',
    };
    return colorMap[domainId] || 'blue';
  };

  const getAgentRolesForMethod = (method) => {
    const roleMap = {
      swarm: ['scout', 'gatherer', 'mutator', 'selector'],
      evolution: ['scout', 'mutagen', 'crossover'],  // scout generates initial population
      cosmic: ['scout', 'exploder', 'cooler', 'synthesizer'],  // scout generates seed
      quantum: ['gauge', 'fermion', 'higgs', 'yukawa', 'potential'],
    };
    return roleMap[method] || [];
  };

  const roleIconMap = {
    scout: FiCompass,
    gatherer: FiArchive,
    mutator: FiRefreshCw,
    selector: FiCheckCircle,
    mutagen: FiZap,
    crossover: FiShuffle,
    exploder: FiSun,
    cooler: FiThermometer,
    synthesizer: FiSliders,
    gauge: FiTarget,
    fermion: FiCpu,
    yukawa: FiLink,
    higgs: FiStar,
    potential: FiTrendingUp,
  };

  // Model recommendations for each agent role
  const roleModelHints = {
    // Swarm agents
    scout: '14-40B creative',
    gatherer: '7-14B analytical',
    mutator: '3-7B fast',
    selector: '14-40B smart',
    // Evolution agents
    mutagen: '7-14B creative',
    crossover: '14-40B reasoning',
    // Cosmic agents
    exploder: '14-40B creative',
    cooler: '3-7B fast',
    synthesizer: '14-40B quality',
    // Quantum agents
    gauge: '14-40B diverse',
    fermion: '3-7B fast',
    higgs: '14-40B analytical',
    yukawa: '3-7B fast',
    potential: 'any (formatter)',
  };

  const getAgentIcon = (role) => roleIconMap[role] || FiCpu;

  // Render domain selection step
  const renderDomainSelection = () => {
    if (isLoadingDomains) {
      return (
        <Box textAlign="center" py={10}>
          <Text>Loading domains...</Text>
        </Box>
      );
    }

    if (isDomainsError) {
      return (
        <Box textAlign="center" py={10}>
          <Alert status="error" flexDirection="column" alignItems="center" justifyContent="center" textAlign="center" borderRadius="md">
            <AlertIcon boxSize="40px" mr={0} />
            <Heading size="md" mt={4} mb={1}>
              Failed to load domains
            </Heading>
            <Text mb={4}>
              {domainsError?.message || 'Could not connect to the server.'}
            </Text>
            <Button colorScheme="red" onClick={() => refetchDomains()}>
              Retry
            </Button>
          </Alert>
        </Box>
      );
    }

    return (
      <Box>
        <Text color="gray.600" mb={4}>
          Select a domain for your dataset:
        </Text>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {domainsData?.domains?.map((domain) => (
            <DomainCard
              key={domain.key}
              domain={domain}
              onClick={() => handleDomainSelect(domain)}
            />
          ))}
        </SimpleGrid>
      </Box>
    );
  };

  // Render subdomain selection step
  const renderSubdomainSelection = () => {
    if (!selectedDomain) return null;

    const domainColor = getDomainColorScheme(selectedDomain.key);

    return (
      <Box>
        <HStack mb={4}>
          <Button size="sm" onClick={() => setCurrentStep(0)}>
            Back
          </Button>
          <Text fontSize="lg">
            {selectedDomain.name} &gt; Select a subdomain
          </Text>
        </HStack>

        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
          {Object.entries(selectedDomain.subdomains).map(([key, subdomain]) => (
            <Card
              key={key}
              cursor="pointer"
              bg={selectedSubdomain?.key === key ? `${domainColor}.50` : cardBg}
              borderWidth="1px"
              borderColor={selectedSubdomain?.key === key ? `${domainColor}.500` : borderColor}
              borderRadius="lg"
              overflow="hidden"
              onClick={() => handleSubdomainSelect({ key, ...subdomain })}
              _hover={{ borderColor: `${domainColor}.300` }}
            >
              <CardHeader>
                <HStack spacing={2}>
                  <Box w={1} h={6} bg={`${domainColor}.500`} borderRadius="full" />
                  <Heading size="md">{subdomain.name}</Heading>
                </HStack>
              </CardHeader>
              <CardBody>
                <Text>{subdomain.description}</Text>
                <Box mt={2}>
                  {subdomain.scenarios.map((scenario) => (
                    <Badge key={scenario} mr={1} mb={1} colorScheme={domainColor}>
                      {scenario}
                    </Badge>
                  ))}
                </Box>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>

        <Box mt={6} p={6} bg={cardBg} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
          {/* Generation Mode Selector */}
          <Heading size="md" mb={4}>Generation Mode</Heading>
          <RadioGroup
            value={generationParams.generation_mode}
            onChange={(value) => handleParamChange('generation_mode', value)}
            mb={6}
          >
            <Stack direction="row" spacing={4}>
              <Radio value="standard">
                <VStack align="start" spacing={0}>
                  <Text fontWeight="bold">Standard (Fast & Simple)</Text>
                  <Text fontSize="sm" color="gray.600">Traditional single-model generation</Text>
                </VStack>
              </Radio>
              <Radio value="advanced">
                <VStack align="start" spacing={0}>
                  <Text fontWeight="bold">Advanced (Multi-Agent)</Text>
                  <Text fontSize="sm" color="gray.600">Multiple LLM agents collaborate, each with its own model</Text>
                </VStack>
              </Radio>
            </Stack>
          </RadioGroup>

          {/* Advanced Method Selection */}
          {generationParams.generation_mode === 'advanced' && (
            <Box mb={6}>
              <Heading size="sm" mb={3}>Select Method</Heading>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3}>
                {[
                  {
                    id: 'swarm',
                    name: 'Ensemble Cascade',
                    desc: 'Scout generates, Gatherer critiques, Mutator varies, Selector curates. Colony-style collaboration.',
                    agents: '4 agents',
                    tooltip: 'Inspired by bee/ant colonies. Scout explores and generates examples, Gatherer evaluates quality, Mutator creates variations (rephrase, formalize, simplify), Selector ensures diversity. Best for: diverse datasets with quality filtering.'
                  },
                  {
                    id: 'evolution',
                    name: 'Swarm Collective Convergence',
                    desc: 'Breeds generations through mutation and crossover, combining best traits. Survival of the fittest.',
                    agents: '3 agents',
                    tooltip: 'Inspired by genetic algorithms. Scout creates initial population, Mutagen applies mutations (complexity, edge cases), Crossover combines best traits from pairs. Examples improve over generations. Best for: iterative refinement.'
                  },
                  {
                    id: 'cosmic',
                    name: 'Evolutionary Agents Fusion',
                    desc: 'Explodes seed into variations, cools and stabilizes, then synthesizes. Big Bang expansion.',
                    agents: '4 agents',
                    tooltip: 'Inspired by Big Bang. Scout creates seed, Exploder expands into many directions (formal, casual, technical, simple), Cooler refines unstable variations, Synthesizer polishes. Best for: creative expansion from one good example.'
                  },
                  {
                    id: 'quantum',
                    name: 'Lattice Network Sync',
                    desc: 'Creates multiple approaches simultaneously, then collapses to the best. Quantum selection.',
                    agents: '5 agents',
                    tooltip: 'Inspired by quantum physics. Gauge creates N approaches (superposition), Fermion generates Q&A for each, Higgs evaluates and selects best (collapse), Yukawa enhances with improvements, Potential formats output. Best for: high-quality through selection.'
                  },
                ].map(method => (
                  <Card
                    key={method.id}
                    cursor="pointer"
                    bg={generationParams.advanced_method === method.id ? `${domainColor}.50` : cardBg}
                    borderWidth="2px"
                    borderColor={generationParams.advanced_method === method.id ? `${domainColor}.500` : borderColor}
                    onClick={() => handleParamChange('advanced_method', method.id)}
                    _hover={{ borderColor: `${domainColor}.300` }}
                  >
                    <CardBody>
                      <HStack justify="space-between" mb={1}>
                        <Text fontWeight="bold">{method.name}</Text>
                        <Tooltip
                          label={method.tooltip}
                          placement="top"
                          hasArrow
                          bg="gray.700"
                          color="white"
                          p={3}
                          borderRadius="md"
                          fontSize="sm"
                          maxW="300px"
                        >
                          <IconButton
                            icon={<FiHelpCircle />}
                            size="xs"
                            variant="ghost"
                            aria-label="Method info"
                            onClick={(e) => e.stopPropagation()}
                          />
                        </Tooltip>
                      </HStack>
                      <Text fontSize="sm" color="gray.600">{method.desc}</Text>
                      <Text fontSize="xs" color="gray.500" mt={1}>{method.agents}</Text>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            </Box>
          )}

          {/* Agent Model Configuration (Advanced Mode Only) */}
          {generationParams.generation_mode === 'advanced' && (
            <Accordion allowToggle mb={6}>
              <AccordionItem border="none">
                <AccordionButton
                  bg={cardBg}
                  borderWidth="1px"
                  borderColor={borderColor}
                  borderRadius="md"
                  _hover={{ bg: accordionHoverBg }}
                  _expanded={{ bg: `${domainColor}.50`, borderColor: `${domainColor}.500` }}
                >
                  <Box flex="1" textAlign="left">
                    <HStack>
                      <Text fontWeight="bold">Agent Model Configuration</Text>
                      <Badge colorScheme="purple" size="sm">Optional</Badge>
                      <Tooltip
                        label="Configure different models for different agent roles. By default, all agents use the main model selected below."
                        placement="top"
                        hasArrow
                      >
                        <Box as="span" cursor="help" color="gray.500">
                          <FiHelpCircle size={14} />
                        </Box>
                      </Tooltip>
                    </HStack>
                  </Box>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={4} bg={cardBg} borderWidth="0 1px 1px 1px" borderColor={borderColor} borderRadius="0 0 md md">
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600" mb={3}>
                      Configure models for each agent role in the {generationParams.advanced_method} method:
                    </Text>
                    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                      {getAgentRolesForMethod(generationParams.advanced_method).map(role => (
                        <Card key={role} size="sm" variant="outline">
                          <CardBody>
                            <HStack spacing={2} mb={2} align="center">
                              <Icon as={getAgentIcon(role)} boxSize={4} color="gray.500" />
                              <Text fontWeight="bold" fontSize="sm">
                                {role.charAt(0).toUpperCase() + role.slice(1)}
                              </Text>
                              <Text fontSize="xs" color="gray.400" fontStyle="italic">
                                {roleModelHints[role]}
                              </Text>
                            </HStack>
                            <VStack spacing={2}>
                              <FormControl size="sm">
                                <FormLabel fontSize="xs">Provider</FormLabel>
                                <Select
                                  size="sm"
                                  value={agentModels[role]?.provider || 'ollama'}
                                  onChange={(e) => setAgentModels({
                                    ...agentModels,
                                    [role]: { ...agentModels[role], provider: e.target.value, model: '' }
                                  })}
                                >
                                  {providersData?.providers && Array.isArray(providersData.providers) && providersData.providers.map(p => (
                                    <option key={p.id} value={p.id} disabled={!p.available}>
                                      {p.name} {!p.available && '(API key required)'}
                                    </option>
                                  ))}
                                </Select>
                              </FormControl>
                              <FormControl size="sm">
                                <FormLabel fontSize="xs">Model</FormLabel>
                                <Select
                                  size="sm"
                                  value={agentModels[role]?.model || ''}
                                  onChange={(e) => setAgentModels({
                                    ...agentModels,
                                    [role]: { ...agentModels[role], model: e.target.value }
                                  })}
                                >
                                  <option value="">Use default</option>
                                  {providerModels[agentModels[role]?.provider] && Array.isArray(providerModels[agentModels[role]?.provider]) &&
                                    providerModels[agentModels[role]?.provider].map(m => (
                                      <option key={m.id} value={m.id}>{m.name}</option>
                                    ))
                                  }
                                </Select>
                              </FormControl>
                            </VStack>
                          </CardBody>
                        </Card>
                      ))}
                    </SimpleGrid>
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          )}

          <Divider my={4} />

          <Heading size="md" mb={4}>Generation Parameters</Heading>

          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <FormControl>
              <HStack mb={2}>
                <FormLabel mb={0}>Output Format</FormLabel>
                <Tooltip
                  label="Chat: Multi-turn conversations (user ↔ assistant). Best for: Customer Support, Tutoring, Dialogue. Instruction: Single-turn tasks (instruction → response). Best for: Q&A, Commands, Documentation."
                  placement="top"
                  hasArrow
                  bg="gray.700"
                  color="white"
                  p={3}
                  borderRadius="md"
                  fontSize="sm"
                  maxW="350px"
                >
                  <Box as="span" cursor="help" color="gray.500">
                    <FiHelpCircle size={14} />
                  </Box>
                </Tooltip>
              </HStack>
              <RadioGroup
                value={generationParams.format}
                onChange={(value) => handleParamChange('format', value)}
              >
                <Stack direction="row">
                  <Radio value="chat">Chat</Radio>
                  <Radio value="instruction">Instruction</Radio>
                </Stack>
              </RadioGroup>
            </FormControl>

            <FormControl>
              <FormLabel>Language</FormLabel>
              <Select
                value={generationParams.language}
                onChange={(e) => handleParamChange('language', e.target.value)}
              >
                <option value="en">English</option>
                <option value="ru">Russian</option>
              </Select>
            </FormControl>

            <FormControl>
              <FormLabel>Number of Examples</FormLabel>
              <Tooltip label="How many examples to generate. More examples = better model, but higher cost" placement="top">
                <NumberInput
                  value={generationParams.count}
                  onChange={(value) => handleParamChange('count', Number(value))}
                  min={1}
                  max={1000}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </Tooltip>
            </FormControl>

            <FormControl>
              <FormLabel>Temperature</FormLabel>
              <Tooltip label="Controls randomness. Lower = more focused, Higher = more creative. Recommended: 0.7" placement="top">
                <Slider
                  value={generationParams.temperature}
                  onChange={(value) => handleParamChange('temperature', value)}
                  min={0}
                  max={1}
                  step={0.1}
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb boxSize={6} />
                </Slider>
              </Tooltip>
            </FormControl>

            <FormControl>
              <FormLabel>Provider</FormLabel>
              <Select
                value={generationParams.provider}
                onChange={(e) => handleParamChange('provider', e.target.value)}
              >
                {providersData?.providers?.map((provider) => (
                  <option
                    key={provider.id}
                    value={provider.id}
                    disabled={!provider.available}
                  >
                    {provider.name} {!provider.available && '(API key required)'}
                  </option>
                ))}
              </Select>
              {providersData?.providers?.find(p => p.id === generationParams.provider && !p.available) && (
                <Text fontSize="sm" color="orange.500" mt={1}>
                  ⚠️ API key not configured for this provider
                </Text>
              )}
            </FormControl>

            <FormControl>
              <FormLabel>Model</FormLabel>
              <Select
                value={generationParams.model}
                onChange={(e) => handleParamChange('model', e.target.value)}
                isDisabled={!generationParams.provider}
              >
                <option value="">Select a model</option>
                {providerModels[generationParams.provider]?.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </Select>
              {modelsData?.models?.find(m => (m.id || m) === generationParams.model)?.description && (
                <Text fontSize="sm" color="gray.600" mt={1}>
                  {modelsData.models.find(m => (m.id || m) === generationParams.model).description}
                </Text>
              )}
            </FormControl>

            <FormControl>
              <FormLabel>
                Prompt Template (Optional)
                <Tooltip label="Manage all templates">
                  <IconButton
                    icon={<FiSettings />}
                    size="xs"
                    ml={2}
                    onClick={onOpen}
                    aria-label="Manage templates"
                    variant="ghost"
                  />
                </Tooltip>
              </FormLabel>

              <Menu closeOnSelect={true} matchWidth>
                <MenuButton
                  as={Button}
                  rightIcon={<FiChevronDown />}
                  leftIcon={selectedTemplate ? <FiFileText /> : null}
                  variant="outline"
                  w="100%"
                  textAlign="left"
                  fontWeight="normal"
                  isLoading={isLoadingTemplates}
                >
                  <Text noOfLines={1}>
                    {selectedTemplate ? selectedTemplate.name : 'Use default template'}
                  </Text>
                </MenuButton>
                <MenuList maxH="400px" overflowY="auto" zIndex={1000}>
                  {/* Search input */}
                  <Box px={3} pb={2} position="sticky" top={0} bg="white" zIndex={1}>
                    <InputGroup size="sm">
                      <InputLeftElement pointerEvents="none">
                        <FiSearch color="gray.400" />
                      </InputLeftElement>
                      <Input
                        placeholder="Search templates..."
                        value={templateSearch}
                        onChange={(e) => setTemplateSearch(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      {templateSearch && (
                        <IconButton
                          icon={<FiX />}
                          size="xs"
                          variant="ghost"
                          position="absolute"
                          right={1}
                          top="50%"
                          transform="translateY(-50%)"
                          onClick={(e) => {
                            e.stopPropagation();
                            setTemplateSearch('');
                          }}
                          aria-label="Clear search"
                        />
                      )}
                    </InputGroup>
                  </Box>

                  {/* Default option */}
                  <MenuItem
                    onClick={() => handleParamChange('template', '')}
                    bg={!generationParams.template ? 'blue.50' : undefined}
                  >
                    <HStack>
                      <Text>Use default template</Text>
                      {!generationParams.template && <Badge colorScheme="blue" size="sm">Selected</Badge>}
                    </HStack>
                  </MenuItem>

                  <MenuDivider />

                  {/* Grouped templates */}
                  {Object.keys(filteredTemplates).length === 0 ? (
                    <Box px={4} py={2}>
                      <Text color="gray.500" fontSize="sm">
                        {templateSearch ? 'No templates found' : 'No templates available'}
                      </Text>
                    </Box>
                  ) : (
                    Object.entries(filteredTemplates).map(([domain, domainTemplates]) => (
                      <MenuGroup
                        key={domain}
                        title={DOMAIN_LABELS[domain] || domain.charAt(0).toUpperCase() + domain.slice(1)}
                        color="gray.600"
                      >
                        {domainTemplates.map((template) => (
                          <MenuItem
                            key={template.id || template.name}
                            onClick={() => handleParamChange('template', template.id || template.name)}
                            bg={generationParams.template === (template.id || template.name) ? 'blue.50' : undefined}
                            _hover={{ bg: 'gray.100' }}
                          >
                            <VStack align="start" spacing={0} w="100%">
                              <HStack justify="space-between" w="100%">
                                <Text fontWeight="medium" noOfLines={1}>{template.name}</Text>
                                {generationParams.template === (template.id || template.name) && (
                                  <Badge colorScheme="blue" size="sm">Selected</Badge>
                                )}
                              </HStack>
                              {template.description && (
                                <Text fontSize="xs" color="gray.500" noOfLines={1}>
                                  {template.description}
                                </Text>
                              )}
                            </VStack>
                          </MenuItem>
                        ))}
                      </MenuGroup>
                    ))
                  )}
                </MenuList>
              </Menu>

              {/* Show selected template info */}
              {selectedTemplate && (
                <Box mt={2} p={2} bg="gray.50" borderRadius="md" fontSize="sm">
                  <HStack justify="space-between">
                    <Text color="gray.600" noOfLines={2}>
                      {selectedTemplate.description || 'No description'}
                    </Text>
                    <IconButton
                      icon={<FiX />}
                      size="xs"
                      variant="ghost"
                      onClick={() => handleParamChange('template', '')}
                      aria-label="Clear template"
                    />
                  </HStack>
                </Box>
              )}

              {!selectedTemplate && (
                <Text fontSize="xs" color="gray.500" mt={1}>
                  {templatesData.length > 0
                    ? `${templatesData.length} templates available`
                    : 'Custom templates allow you to control generation prompts'}
                </Text>
              )}
            </FormControl>
          </SimpleGrid>

          <HStack mt={8} spacing={4}>
            <Button
              colorScheme="gray"
              variant="outline"
              isDisabled={!selectedSubdomain}
              onClick={handleGeneratePreview}
              isLoading={isGeneratingPreview}
            >
              Generate Preview (1 Example)
            </Button>

            <Button
              colorScheme="blue"
              isDisabled={!selectedSubdomain}
              onClick={handleStartGeneration}
              isLoading={startGenerationMutation.isLoading}
            >
              Start Full Generation
            </Button>
          </HStack>
        </Box>
      </Box >
    );
  };

  // Render generation progress step
  const renderGenerationProgress = () => {
    if (!jobId) return null;

    const isAdvancedMode = generationParams.generation_mode === 'advanced';

    return (
      <Box>
        <HStack mb={4}>
          <Button size="sm" onClick={handleReset}>
            Start New Generation
          </Button>
          <Text fontSize="lg">Generation in Progress</Text>
          {isAdvancedMode && (
            <Badge colorScheme="purple" ml={2}>
              Advanced Mode: {generationParams.advanced_method}
            </Badge>
          )}
        </HStack>

        <SimpleGrid columns={{ base: 1, lg: isAdvancedMode ? 2 : 1 }} spacing={6}>
          {/* Main Progress Card */}
          <Card p={6}>
            <VStack spacing={4} align="stretch">
              <Heading size="lg">
                Generating {selectedDomain?.name} &gt; {selectedSubdomain?.name} Dataset
              </Heading>

              <SimpleGrid columns={2} spacing={4}>
                <Stat>
                  <StatLabel>Format</StatLabel>
                  <StatNumber fontSize="md">
                    {generationParams.format === 'chat' ? 'Chat' : 'Instruction'}
                  </StatNumber>
                </Stat>
                <Stat>
                  <StatLabel>Language</StatLabel>
                  <StatNumber fontSize="md">
                    {generationParams.language === 'en' ? 'English' : 'Russian'}
                  </StatNumber>
                </Stat>
                <Stat>
                  <StatLabel>Job ID</StatLabel>
                  <StatNumber fontSize="md">{jobId}</StatNumber>
                </Stat>
                <Stat>
                  <StatLabel>Status</StatLabel>
                  <StatNumber fontSize="md">
                    <Badge
                      colorScheme={
                        jobStatus?.status === 'completed' ? 'green' :
                          jobStatus?.status === 'failed' ? 'red' :
                            jobStatus?.status === 'running' ? 'blue' : 'gray'
                      }
                      fontSize="sm"
                    >
                      {jobStatus?.status || 'Pending'}
                    </Badge>
                  </StatNumber>
                </Stat>
              </SimpleGrid>

              <Divider />

              <Text fontWeight="bold">
                Progress: {jobStatus?.examples_generated || 0} / {jobStatus?.examples_requested || generationParams.count} examples
              </Text>

              <Progress
                value={calculateProgress()}
                size="lg"
                colorScheme={jobStatus?.status === 'completed' ? 'green' : 'blue'}
                borderRadius="md"
                hasStripe={jobStatus?.status === 'running'}
                isAnimated={jobStatus?.status === 'running'}
              />

              {jobStatus?.status === 'completed' && jobStatus?.dataset_id && (
                <Button
                  as={Link}
                  to={`/datasets/${jobStatus.dataset_id}`}
                  colorScheme="green"
                >
                  View Generated Dataset
                </Button>
              )}

              {jobStatus?.status === 'failed' && (
                <Alert status="error" borderRadius="md">
                  <AlertIcon />
                  <Box>
                    <Text fontWeight="bold">Generation failed</Text>
                    <Text fontSize="sm">{jobStatus?.errors?.join(', ') || 'Unknown error'}</Text>
                  </Box>
                </Alert>
              )}
            </VStack>
          </Card>

          {/* Agent Monitor for Advanced Mode */}
          {isAdvancedMode && (
            <Card p={6}>
              <CardHeader p={0} mb={4}>
                <Heading size="md">🤖 Agent Activity Monitor</Heading>
                <Text fontSize="sm" color="gray.500">
                  Real-time view of multi-agent generation
                </Text>
              </CardHeader>
              <CardBody p={0}>
                <AgentMonitor
                  jobId={jobId}
                  onComplete={(data) => {
                    toast({
                      title: 'Generation Complete',
                      description: `Generated ${data?.examples_count || 0} examples`,
                      status: 'success',
                      duration: 5000,
                    });
                  }}
                />
              </CardBody>
            </Card>
          )}
        </SimpleGrid>
      </Box>
    );
  };

  // Render the current step
  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return renderDomainSelection();
      case 1:
        return renderSubdomainSelection();
      case 2:
        return renderGenerationProgress();
      default:
        return null;
    }
  };

  return (
    <Box>
      <Heading size="md" mb={6}>Dataset Generator</Heading>

      {renderCurrentStep()}

      {/* Template Management Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="6xl">
        <ModalOverlay />
        <ModalContent maxH="90vh">
          <ModalHeader>
            {selectedTemplate ? `Template: ${selectedTemplate.name}` : 'Custom Prompt Template'}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6} overflowY="auto">
            <VStack align="stretch" spacing={4}>
              {selectedTemplate ? (
                <>
                  <HStack justify="space-between" align="start">
                    <VStack align="start" spacing={1}>
                      <Badge colorScheme="blue">{selectedTemplate.domain}</Badge>
                      {selectedTemplate.subdomain && (
                        <Badge colorScheme="purple">{selectedTemplate.subdomain}</Badge>
                      )}
                    </VStack>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        setCustomTemplate(selectedTemplate.content || '');
                        handleParamChange('template', '');
                      }}
                    >
                      Copy to Custom
                    </Button>
                  </HStack>

                  {selectedTemplate.description && (
                    <Text fontSize="sm" color="gray.600">
                      {selectedTemplate.description}
                    </Text>
                  )}

                  <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden" h="400px">
                    <Editor
                      height="400px"
                      defaultLanguage="markdown"
                      value={selectedTemplate.content || ''}
                      theme="vs-light"
                      options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        wordWrap: 'on',
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        readOnly: true,
                      }}
                    />
                  </Box>

                  <Text fontSize="xs" color="gray.500">
                    This is a saved template. To modify it, copy to custom or go to Templates page.
                  </Text>
                </>
              ) : (
                <>
                  <Text fontSize="sm" color="gray.600">
                    Customize the prompt template for {selectedSubdomain?.name || 'this domain'}.
                    Use variables like {'{domain}'}, {'{subdomain}'}, {'{format}'} for dynamic content.
                  </Text>

                  <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden" h="400px">
                    <Editor
                      height="400px"
                      defaultLanguage="markdown"
                      value={customTemplate}
                      onChange={(value) => setCustomTemplate(value || '')}
                      theme="vs-light"
                      options={{
                        minimap: { enabled: false },
                        fontSize: 14,
                        wordWrap: 'on',
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                      }}
                    />
                  </Box>

                  <Text fontSize="xs" color="gray.500">
                    💡 Tip: Use clear instructions and examples for better results
                  </Text>
                </>
              )}
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>
              Close
            </Button>
            {!selectedTemplate && customTemplate && (
              <Button colorScheme="blue" onClick={onClose}>
                Apply Custom Template
              </Button>
            )}
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* Preview Modal */}
      <Modal isOpen={isPreviewOpen} onClose={onPreviewClose} size="xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Preview Example</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {previewExample ? (
              <VStack align="stretch" spacing={4}>
                <Text fontWeight="bold">Generated Example:</Text>
                {previewExample.messages ? (
                  <VStack align="stretch" spacing={2}>
                    {previewExample.messages.map((msg, idx) => (
                      <Box
                        key={idx}
                        p={3}
                        bg={msg.role === 'user' ? 'blue.50' : 'green.50'}
                        borderRadius="md"
                      >
                        <Text fontSize="xs" fontWeight="bold" mb={1}>
                          {msg.role.toUpperCase()}
                        </Text>
                        <Text>{msg.content}</Text>
                      </Box>
                    ))}
                  </VStack>
                ) : previewExample.instruction ? (
                  <VStack align="stretch" spacing={2}>
                    <Box p={3} bg="blue.50" borderRadius="md">
                      <Text fontSize="xs" fontWeight="bold" mb={1}>INSTRUCTION</Text>
                      <Text>{previewExample.instruction}</Text>
                    </Box>
                    <Box p={3} bg="green.50" borderRadius="md">
                      <Text fontSize="xs" fontWeight="bold" mb={1}>OUTPUT</Text>
                      <Text>{previewExample.output}</Text>
                    </Box>
                  </VStack>
                ) : (
                  <Code p={4} borderRadius="md" whiteSpace="pre-wrap">
                    {JSON.stringify(previewExample, null, 2)}
                  </Code>
                )}

                <Divider />

                <HStack justify="space-between">
                  <Button variant="outline" onClick={onPreviewClose}>
                    Cancel
                  </Button>
                  <Button
                    colorScheme="blue"
                    onClick={() => {
                      onPreviewClose();
                      handleStartGeneration();
                    }}
                  >
                    Looks Good - Start Full Generation
                  </Button>
                </HStack>
              </VStack>
            ) : (
              <Text>No preview available</Text>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default Generator;