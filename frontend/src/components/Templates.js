import React, { useState } from 'react';
import {
    Box,
    Heading,
    Text,
    SimpleGrid,
    Card,
    CardHeader,
    CardBody,
    CardFooter,
    Button,
    Badge,
    HStack,
    VStack,
    IconButton,
    useColorModeValue,
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    useDisclosure,
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    ModalCloseButton,
    FormControl,
    FormLabel,
    Input,
    Select,
    Textarea,
    useToast,
    Tabs,
    TabList,
    Tab,
} from '@chakra-ui/react';
import { FiMoreVertical, FiPlus, FiEdit, FiTrash2, FiCopy, FiCode, FiClock, FiRotateCcw } from 'react-icons/fi';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import Editor from '@monaco-editor/react';
import { getTemplates, createTemplate, updateTemplate, deleteTemplate, getTemplateVersions, restoreTemplateVersion, getDomains } from '../services/api';
const TEMPLATE_TABS = [
    { key: 'all', label: 'All Templates' },
    {
        key: 'business_ops',
        label: 'Business & Operations',
        domains: ['business', 'sales', 'financial', 'marketing', 'hr', 'ecommerce', 'meetings'],
    },
    {
        key: 'professional_services',
        label: 'Professional Services',
        domains: ['medical', 'legal', 'education', 'coaching', 'research', 'data'],
    },
    {
        key: 'creative_general',
        label: 'Creative & General',
        domains: ['creative', 'gaming', 'general', 'support'],
    },
];

const DOMAIN_LABELS = {
    general: 'General Assistant',
    support: 'Customer Support',
    medical: 'Medical Information',
    legal: 'Legal Documentation',
    education: 'Educational Content',
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
};

const DOMAIN_OPTIONS = Object.entries(DOMAIN_LABELS).map(([key, name]) => ({ key, name }));
const ALLOWED_DOMAINS = new Set(Object.keys(DOMAIN_LABELS));

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
        data: 'teal',
        general: 'purple',
    };
    return colorMap[domainId] || 'purple';
};

const Templates = () => {
    const [selectedTab, setSelectedTab] = useState('all');
    const [editingTemplate, setEditingTemplate] = useState(null);
    const [prefillData, setPrefillData] = useState(null);
    const { isOpen, onOpen, onClose } = useDisclosure();
    const toast = useToast();
    const queryClient = useQueryClient();

    const cardBg = useColorModeValue('white', 'gray.700');
    const borderColor = useColorModeValue('gray.200', 'gray.600');

    // Fetch templates
    const { data: templates = [], isLoading: isLoadingTemplates } = useQuery({
        queryKey: ['templates'],
        queryFn: () => getTemplates(),
    });

    // Fetch domain metadata for subdomain coverage
    const { data: domainsData, isLoading: isLoadingDomains } = useQuery({
        queryKey: ['domains'],
        queryFn: getDomains,
    });

    const handleModalClose = () => {
        setPrefillData(null);
        onClose();
    };

    const allowedTemplates = React.useMemo(
        () => templates.filter(template => ALLOWED_DOMAINS.has(template.domain)),
        [templates]
    );

    const subdomainLabelMap = React.useMemo(() => {
        if (!domainsData?.domains) return {};
        const map = {};
        const domainList = Array.isArray(domainsData.domains) ? domainsData.domains : Object.values(domainsData.domains || {});
        domainList.forEach(domain => {
            if (!ALLOWED_DOMAINS.has(domain.key)) return;
            Object.entries(domain.subdomains || {}).forEach(([subKey, subdomain]) => {
                map[`${domain.key}:${subKey}`] = subdomain.name;
            });
        });
        return map;
    }, [domainsData]);

    const placeholderTemplates = React.useMemo(() => {
        if (!domainsData?.domains) return [];
        const domainList = Array.isArray(domainsData.domains) ? domainsData.domains : Object.values(domainsData.domains || {});
        const placeholders = [];
        domainList.forEach(domain => {
            if (!ALLOWED_DOMAINS.has(domain.key)) return;
            Object.entries(domain.subdomains || {}).forEach(([subKey, subdomain]) => {
                const hasTemplate = allowedTemplates.some(template => template.domain === domain.key && (template.subdomain || '') === subKey);
                if (!hasTemplate) {
                    const scenarios = (subdomain.scenarios || []).slice(0, 3).join(', ');
                    placeholders.push({
                        id: `placeholder-${domain.key}-${subKey}`,
                        name: `${subdomain.name} Example`,
                        domain: domain.key,
                        subdomain: subKey,
                        description: subdomain.description || `Auto-generated example for ${subdomain.name}.`,
                        content: `You are generating data for the ${subdomain.name} subdomain within ${domain.name}. Focus on scenarios like ${scenarios || 'the key workflows described in the requirements'}.

User Context: {{user_context}}
Goal: {{goal}}

Respond with a structured example that showcases best practices for this subdomain.`,
                        variables: {},
                        updated_at: '1970-01-01T00:00:00.000Z',
                        isPlaceholder: true,
                        subdomainDisplayName: subdomain.name,
                    });
                }
            });
        });
        return placeholders;
    }, [domainsData, allowedTemplates]);

    const templatesWithExamples = React.useMemo(
        () => [...allowedTemplates, ...placeholderTemplates],
        [allowedTemplates, placeholderTemplates]
    );

    const filteredTemplates = React.useMemo(() => {
        if (selectedTab === 'all') return templatesWithExamples;
        const tabConfig = TEMPLATE_TABS.find(tab => tab.key === selectedTab);
        if (!tabConfig?.domains) return templatesWithExamples;
        return templatesWithExamples.filter(template => tabConfig.domains.includes(template.domain));
    }, [selectedTab, templatesWithExamples]);

    const currentTabIndex = Math.max(0, TEMPLATE_TABS.findIndex(tab => tab.key === selectedTab));

    const isLoading = isLoadingTemplates || isLoadingDomains;

    // Create mutation
    const createMutation = useMutation({
        mutationFn: createTemplate,
        onSuccess: () => {
            queryClient.invalidateQueries(['templates']);
            toast({ title: 'Template created', status: 'success' });
            handleModalClose();
        },
    });

    // Update mutation
    const updateMutation = useMutation({
        mutationFn: ({ id, data }) => updateTemplate(id, data),
        onSuccess: () => {
            queryClient.invalidateQueries(['templates']);
            toast({ title: 'Template updated', status: 'success' });
            handleModalClose();
        },
    });

    // Delete mutation
    const deleteMutation = useMutation({
        mutationFn: deleteTemplate,
        onSuccess: () => {
            queryClient.invalidateQueries(['templates']);
            toast({ title: 'Template deleted', status: 'success' });
        },
    });

    // Restore mutation
    const restoreMutation = useMutation({
        mutationFn: ({ templateId, versionId }) => restoreTemplateVersion(templateId, versionId),
        onSuccess: () => {
            queryClient.invalidateQueries(['templates']);
            toast({ title: 'Version restored', status: 'success' });
        },
    });

    const handleSave = (formData) => {
        if (editingTemplate) {
            updateMutation.mutate({ id: editingTemplate.id, data: formData });
        } else {
            createMutation.mutate(formData);
        }
    };

    const handleEdit = (template) => {
        setPrefillData(null);
        setEditingTemplate(template);
        onOpen();
    };

    const handleDelete = (id) => {
        if (window.confirm('Are you sure you want to delete this template?')) {
            deleteMutation.mutate(id);
        }
    };

    const handleDuplicate = (template) => {
        const { id, created_at, updated_at, ...rest } = template;
        createMutation.mutate({
            ...rest,
            name: `${template.name} (Copy)`,
        });
    };

    const handleRestore = async (templateId, versionId) => {
        if (window.confirm('Are you sure you want to restore this version? Current changes will be saved as a new version.')) {
            try {
                const restored = await restoreMutation.mutateAsync({ templateId, versionId });
                return restored;
            } catch (error) {
                toast({ title: 'Failed to restore', status: 'error' });
                return null;
            }
        }
        return null;
    };

    const openCreateModal = () => {
        setPrefillData(null);
        setEditingTemplate(null);
        onOpen();
    };

    const openPlaceholderModal = (template) => {
        const { id, isPlaceholder, subdomainDisplayName, updated_at, ...rest } = template;
        setPrefillData({
            ...rest,
            name: `${subdomainDisplayName || template.subdomain || 'New'} Template`,
        });
        setEditingTemplate(null);
        onOpen();
    };

    return (
        <Box>
            <Heading size="md" mb={6}>Prompt Templates</Heading>

            <HStack justify="space-between" mb={6}>
                <Text color="gray.600">Manage and customize your generation templates.</Text>
                <Button leftIcon={<FiPlus />} colorScheme="blue" onClick={openCreateModal}>
                    New Template
                </Button>
            </HStack>

            <Tabs
                variant="line"
                colorScheme="blue"
                mb={6}
                index={currentTabIndex}
                onChange={(index) => {
                    const tabKey = TEMPLATE_TABS[index]?.key || 'all';
                    setSelectedTab(tabKey);
                }}
            >
                <TabList mb={4}>
                    {TEMPLATE_TABS.map(tab => (
                        <Tab key={tab.key}>{tab.label}</Tab>
                    ))}
                </TabList>
            </Tabs>

            {isLoading ? (
                <Text color="gray.500">Loading templates...</Text>
            ) : filteredTemplates.length === 0 ? (
                <Text color="gray.500">No templates in this category yet.</Text>
            ) : (
                <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                    {filteredTemplates.map((template) => (
                        <Card key={template.id} bg={cardBg} borderWidth="1px" borderColor={borderColor} _hover={{ shadow: 'md' }}>
                            <CardHeader pb={2}>
                                <HStack justify="space-between" align="start">
                                    <VStack align="start" spacing={1}>
                                        <Heading size="md">{template.name}</Heading>
                                        <HStack spacing={2} align="center">
                                            <Badge colorScheme={getDomainColorScheme(template.domain)}>
                                                {DOMAIN_LABELS[template.domain] || template.domain}
                                            </Badge>
                                            {template.subdomain && (
                                                <Badge colorScheme={getDomainColorScheme(template.domain)}>
                                                    {subdomainLabelMap[`${template.domain}:${template.subdomain}`] || template.subdomain}
                                                </Badge>
                                            )}
                                        </HStack>
                                    </VStack>
                                    <Menu>
                                        <MenuButton as={IconButton} icon={<FiMoreVertical />} variant="ghost" size="sm" />
                                        <MenuList>
                                            {template.isPlaceholder ? (
                                                <MenuItem icon={<FiPlus />} onClick={() => openPlaceholderModal(template)}>Create Template</MenuItem>
                                            ) : (
                                                <>
                                                    <MenuItem icon={<FiEdit />} onClick={() => handleEdit(template)}>Edit</MenuItem>
                                                    <MenuItem icon={<FiCopy />} onClick={() => handleDuplicate(template)}>Duplicate</MenuItem>
                                                    <MenuItem icon={<FiTrash2 />} color="red.500" onClick={() => handleDelete(template.id)}>Delete</MenuItem>
                                                </>
                                            )}
                                        </MenuList>
                                    </Menu>
                                </HStack>
                            </CardHeader>
                            <CardBody py={2}>
                                <Text fontSize="sm" color="gray.600" noOfLines={2} mb={3}>
                                    {template.description || 'No description provided.'}
                                </Text>
                                <Box bg="gray.50" p={2} borderRadius="md" fontSize="xs" fontFamily="monospace" noOfLines={3}>
                                    {template.content}
                                </Box>
                            </CardBody>
                            <CardFooter pt={2}>
                                <Text fontSize="xs" color="gray.500">
                                    Updated: {new Date(template.updated_at).toLocaleDateString()}
                                </Text>
                            </CardFooter>
                        </Card>
                    ))}
                </SimpleGrid>
            )}

            <TemplateModal
                isOpen={isOpen}
                onClose={handleModalClose}
                initialData={editingTemplate}
                prefillData={prefillData}
                onSave={handleSave}
                onRestore={handleRestore}
            />
        </Box>
    );
};

const TemplateModal = ({ isOpen, onClose, initialData, prefillData, onSave, onRestore }) => {
    const [activeTab, setActiveTab] = useState(0);
    const [versions, setVersions] = useState([]);
    const [isLoadingVersions, setIsLoadingVersions] = useState(false);

    const [formData, setFormData] = useState({
        name: '',
        domain: 'general',
        subdomain: '',
        description: '',
        content: '',
        variables: {},
    });

    const [detectedVariables, setDetectedVariables] = useState([]);

    // Detect variables when content changes
    React.useEffect(() => {
        if (formData.content) {
            const regex = /{{([^}]+)}}/g;
            const matches = [...formData.content.matchAll(regex)];
            const variables = [...new Set(matches.map(m => m[1].trim()))];
            setDetectedVariables(variables);

            // Preserve existing variable config, add new ones
            setFormData(prev => {
                const newVariables = { ...prev.variables };
                variables.forEach(v => {
                    if (!newVariables[v]) {
                        newVariables[v] = { default: '', description: '' };
                    }
                });
                return { ...prev, variables: newVariables };
            });
        } else {
            setDetectedVariables([]);
        }
    }, [formData.content]);

    // Initialize form data
    React.useEffect(() => {
        if (initialData) {
            setFormData({
                ...initialData,
                variables: initialData.variables || {}
            });
        } else if (prefillData) {
            setFormData({
                name: prefillData.name || '',
                domain: prefillData.domain || 'general',
                subdomain: prefillData.subdomain || '',
                description: prefillData.description || '',
                content: prefillData.content || '',
                variables: prefillData.variables || {},
            });
        } else {
            setFormData({
                name: '',
                domain: 'general',
                subdomain: '',
                description: '',
                content: '',
                variables: {},
            });
        }
        setActiveTab(0); // Reset to editor tab on open
    }, [initialData, isOpen, prefillData]);

    // Fetch versions when tab changes to History (index 1)
    React.useEffect(() => {
        if (activeTab === 1 && initialData?.id) {
            setIsLoadingVersions(true);
            getTemplateVersions(initialData.id)
                .then(data => setVersions(data))
                .catch(console.error)
                .finally(() => setIsLoadingVersions(false));
        }
    }, [activeTab, initialData]);

    const handleChange = (field, value) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    const handleVariableChange = (varName, field, value) => {
        setFormData(prev => ({
            ...prev,
            variables: {
                ...prev.variables,
                [varName]: {
                    ...prev.variables[varName],
                    [field]: value
                }
            }
        }));
    };

    const handleVersionRestore = async (version) => {
        const restored = await onRestore(initialData.id, version.id);
        if (restored) {
            setFormData(restored);
            setActiveTab(0); // Switch back to editor
        }
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="6xl">
            <ModalOverlay />
            <ModalContent maxH="90vh">
                <ModalHeader>
                    <HStack justify="space-between">
                        <Text>{initialData ? 'Edit Template' : 'New Template'}</Text>
                        {initialData && (
                            <HStack spacing={4} mr={8}>
                                <Button
                                    size="sm"
                                    variant={activeTab === 0 ? "solid" : "ghost"}
                                    colorScheme={activeTab === 0 ? "blue" : "gray"}
                                    onClick={() => setActiveTab(0)}
                                    leftIcon={<FiEdit />}
                                >
                                    Editor
                                </Button>
                                <Button
                                    size="sm"
                                    variant={activeTab === 1 ? "solid" : "ghost"}
                                    colorScheme={activeTab === 1 ? "blue" : "gray"}
                                    onClick={() => setActiveTab(1)}
                                    leftIcon={<FiClock />}
                                >
                                    History
                                </Button>
                            </HStack>
                        )}
                    </HStack>
                </ModalHeader>
                <ModalCloseButton />
                <ModalBody pb={6} overflowY="auto">
                    {activeTab === 0 ? (
                        <SimpleGrid columns={2} spacing={6}>
                            <VStack align="stretch" spacing={4}>
                                <FormControl isRequired>
                                    <FormLabel>Name</FormLabel>
                                    <Input
                                        value={formData.name}
                                        onChange={(e) => handleChange('name', e.target.value)}
                                        placeholder="e.g., Creative Story Writer"
                                    />
                                </FormControl>

                                <SimpleGrid columns={2} spacing={4}>
                                    <FormControl isRequired>
                                        <FormLabel>Domain</FormLabel>
                                        <Select
                                            value={formData.domain}
                                            onChange={(e) => handleChange('domain', e.target.value)}
                                        >
                                            {DOMAIN_OPTIONS.map(d => (
                                                <option key={d.key} value={d.key}>{d.name}</option>
                                            ))}
                                            {!DOMAIN_LABELS[formData.domain] && formData.domain && (
                                                <option value={formData.domain}>{formData.domain}</option>
                                            )}
                                        </Select>
                                    </FormControl>
                                    <FormControl>
                                        <FormLabel>Subdomain (Optional)</FormLabel>
                                        <Input
                                            value={formData.subdomain || ''}
                                            onChange={(e) => handleChange('subdomain', e.target.value)}
                                            placeholder="e.g., sci-fi"
                                        />
                                    </FormControl>
                                </SimpleGrid>

                                <FormControl>
                                    <FormLabel>Description</FormLabel>
                                    <Textarea
                                        value={formData.description}
                                        onChange={(e) => handleChange('description', e.target.value)}
                                        placeholder="Brief description of what this template does"
                                        rows={2}
                                    />
                                </FormControl>

                                {detectedVariables.length > 0 && (
                                    <Box borderWidth="1px" borderRadius="md" p={4} bg="gray.50">
                                        <Text fontWeight="bold" mb={3} fontSize="sm">Detected Variables</Text>
                                        <VStack spacing={3}>
                                            {detectedVariables.map(variable => (
                                                <Box key={variable} w="100%" p={3} bg="white" borderRadius="md" borderWidth="1px">
                                                    <HStack mb={2}>
                                                        <Badge colorScheme="purple">{`{{${variable}}}`}</Badge>
                                                    </HStack>
                                                    <SimpleGrid columns={2} spacing={3}>
                                                        <FormControl size="sm">
                                                            <FormLabel fontSize="xs">Default Value</FormLabel>
                                                            <Input
                                                                size="sm"
                                                                value={formData.variables[variable]?.default || ''}
                                                                onChange={(e) => handleVariableChange(variable, 'default', e.target.value)}
                                                                placeholder="Default value"
                                                            />
                                                        </FormControl>
                                                        <FormControl size="sm">
                                                            <FormLabel fontSize="xs">Description</FormLabel>
                                                            <Input
                                                                size="sm"
                                                                value={formData.variables[variable]?.description || ''}
                                                                onChange={(e) => handleVariableChange(variable, 'description', e.target.value)}
                                                                placeholder="Help text for user"
                                                            />
                                                        </FormControl>
                                                    </SimpleGrid>
                                                </Box>
                                            ))}
                                        </VStack>
                                    </Box>
                                )}
                            </VStack>

                            <VStack align="stretch" spacing={4}>
                                <FormControl isRequired h="100%">
                                    <FormLabel>Template Content</FormLabel>
                                    <Text fontSize="xs" color="gray.500" mb={2}>
                                        Use {'{{variable}}'} syntax to create dynamic fields.
                                    </Text>
                                    <Box border="1px" borderColor="gray.200" borderRadius="md" overflow="hidden" h="500px">
                                        <Editor
                                            height="100%"
                                            defaultLanguage="markdown"
                                            value={formData.content}
                                            onChange={(value) => handleChange('content', value)}
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
                                </FormControl>
                            </VStack>
                        </SimpleGrid>
                    ) : (
                        <VStack align="stretch" spacing={4}>
                            {isLoadingVersions ? (
                                <Text>Loading history...</Text>
                            ) : versions.length === 0 ? (
                                <Text color="gray.500">No version history available.</Text>
                            ) : (
                                versions.map(version => (
                                    <Box key={version.id} p={4} borderWidth="1px" borderRadius="md">
                                        <HStack justify="space-between" mb={2}>
                                            <HStack>
                                                <Badge colorScheme="blue">v{version.version}</Badge>
                                                <Text fontSize="sm" color="gray.500">
                                                    {new Date(version.created_at).toLocaleString()}
                                                </Text>
                                            </HStack>
                                            <Button
                                                size="xs"
                                                leftIcon={<FiRotateCcw />}
                                                onClick={() => handleVersionRestore(version)}
                                            >
                                                Restore
                                            </Button>
                                        </HStack>
                                        <Text fontSize="sm" fontWeight="bold">{version.name}</Text>
                                        <Text fontSize="xs" color="gray.600" noOfLines={2}>
                                            {version.content}
                                        </Text>
                                    </Box>
                                ))
                            )}
                        </VStack>
                    )}
                </ModalBody>
                <ModalFooter justify="flex-end" gap={3} pt={0} pb={6} px={6} bg="white" position="sticky" bottom={0} zIndex={10}>
                    <Button variant="ghost" onClick={onClose}>Cancel</Button>
                    {activeTab === 0 && (
                        <Button colorScheme="blue" onClick={() => onSave(formData)}>
                            Save Template
                        </Button>
                    )}
                </ModalFooter>
            </ModalContent>
        </Modal>
    );
};

export default Templates;
